package auditservice

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	nextRoundInboxPage    = 200
	maxNextRoundInboxScan = 10_000
)

// PrepareNextRound snapshots previously admitted, not-yet-consumed proposal
// checks into immutable Project artifacts. It does not accept the Round: the
// Controller subsequently commits params under its live claim, where exact
// proposal holds and consume-once relations are revalidated atomically.
func (s *Service) PrepareNextRound(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
) (auditstore.AcceptRoundParams, *auditstore.StopReason, error) {
	if snapshot.Round == nil || snapshot.Round.State != auditstore.RoundClosed ||
		claim.AuditID != snapshot.Audit.AuditID || snapshot.Audit.CurrentRoundID == nil ||
		*snapshot.Audit.CurrentRoundID != snapshot.Round.RoundID {
		return auditstore.AcceptRoundParams{}, nil, fmt.Errorf("next Audit Round snapshot is invalid")
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(snapshot.Audit.ProfileSnapshot)
	if err != nil || profile.Ref.Name != snapshot.Audit.Profile.Name ||
		profile.Ref.Version != snapshot.Audit.Profile.Version || profile.Ref.Digest != snapshot.Audit.Profile.Digest {
		return auditstore.AcceptRoundParams{}, nil, fmt.Errorf("pinned AuditProfile cannot build a next Round")
	}
	if profile.Interaction.FindingConfirmation == config.AuditFindingDisabled {
		return auditstore.AcceptRoundParams{}, nil, nil
	}
	baseline, err := DecodeBaseline(snapshot.Audit.BaselineSnapshot)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	selection, err := DecodeDraftSelection(snapshot.Audit.InputSelection)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	var totalItems int
	if err := s.pool.QueryRow(ctx, `SELECT count(*) FROM audit_items WHERE audit_id = $1`,
		snapshot.Audit.AuditID).Scan(&totalItems); err != nil {
		return auditstore.AcceptRoundParams{}, nil, fmt.Errorf("count Audit items: %w", err)
	}
	remainingTotal := snapshot.Audit.Limits.MaxItemsTotal - totalItems
	capacity := snapshot.Audit.Limits.MaxItemsPerRound
	if remainingTotal < capacity {
		capacity = remainingTotal
	}
	proposalSelection, err := s.selectNextProposalChecks(
		ctx, snapshot.Audit, capacity,
	)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	selected := proposalSelection.Proposals
	if len(selected) == 0 {
		switch {
		case proposalSelection.Eligible && snapshot.Round.Ordinal >= snapshot.Audit.Limits.MaxRounds:
			return auditstore.AcceptRoundParams{}, &auditstore.StopReason{
				Code:    "round_budget_exhausted",
				Message: "Unscheduled finding checks remain after the configured Round budget was exhausted.",
			}, nil
		case proposalSelection.Eligible && capacity <= 0:
			return auditstore.AcceptRoundParams{}, &auditstore.StopReason{
				Code:    "item_budget_exhausted",
				Message: "Unscheduled finding checks remain after the configured Audit item budget was exhausted.",
			}, nil
		case proposalSelection.ScanExhausted:
			return auditstore.AcceptRoundParams{}, &auditstore.StopReason{
				Code:    "proposal_scan_budget_exhausted",
				Message: "The bounded finding inbox scan ended before more schedulable work could be established.",
			}, nil
		default:
			return auditstore.AcceptRoundParams{}, nil, nil
		}
	}
	if snapshot.Round.Ordinal >= snapshot.Audit.Limits.MaxRounds {
		return auditstore.AcceptRoundParams{}, &auditstore.StopReason{
			Code:    "round_budget_exhausted",
			Message: "Unscheduled finding checks remain after the configured Round budget was exhausted.",
		}, nil
	}
	if capacity <= 0 {
		return auditstore.AcceptRoundParams{}, &auditstore.StopReason{
			Code:    "item_budget_exhausted",
			Message: "Unscheduled finding checks remain after the configured Audit item budget was exhausted.",
		}, nil
	}

	source := auditdomain.FindingInventoryDocument{
		Schema: auditdomain.FindingInventorySchema, Proposals: selected,
	}
	sourceBytes, err := auditdomain.EncodeFindingInventory(source)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(s.pool)).Project(
		snapshot.Audit.ProjectID,
	)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	namespace := auditdomain.ArtifactNamespace(snapshot.Audit.AuditID)
	sourceArtifact, err := writeImmutableArtifact(
		ctx, projectArtifacts,
		contracts.ArtifactRef{
			Namespace: namespace,
			Name:      auditdomain.DeterministicID("proposal-inventory", snapshot.Audit.AuditID, auditdomain.DigestBytes(sourceBytes)),
		},
		artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: sourceBytes},
	)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	approval := auditdomain.ApprovalNone
	if profile.Interaction.ActiveChecks == config.AuditActiveChecksApprovalRequired &&
		workflowRoleSelectsClassifiedTool(profile, profile.Inventory.ItemWorkflowRole, true) {
		// Proposed-check methods are operator/model data, not a safe classifier.
		// Requiring approval for the whole later-round set is conservative and
		// cannot weaken a profile that requests an active-check gate.
		approval = auditdomain.ApprovalActiveCheck
	}
	roundOrdinal := snapshot.Round.Ordinal + 1
	inventory, err := auditdomain.BuildFindingInventory(sourceBytes, auditdomain.InventoryOptions{
		Round: roundOrdinal, WorkflowRole: profile.Inventory.ItemWorkflowRole,
		SourceInputName: "proposal_inventory", SourceRef: sourceArtifact.Ref,
		Scope: baseline.Scope.Values(), ApprovalRequirement: approval,
	})
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	taskArtifacts, manifest, err := writeTaskPackages(
		ctx, projectArtifacts, namespace, profile, selection, inventory,
	)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	worklist, err := writeRoundPackage(
		ctx, projectArtifacts, namespace, roundOrdinal, inventory, manifest,
	)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	descriptors := make(map[string]auditstore.ExactArtifact, len(selected))
	for _, proposal := range selected {
		descriptors[proposal.ReceiptID] = auditstore.ExactArtifact{
			Ref: proposal.Proposal.Ref, Digest: proposal.Proposal.Digest,
			MediaType: proposal.Proposal.MediaType, SizeBytes: proposal.Proposal.SizeBytes,
		}
	}
	items := make([]auditstore.MaterializedItem, len(inventory.Worklist.Items))
	for index, item := range inventory.Worklist.Items {
		task := inventory.Tasks[index].Document
		if task.Finding == nil {
			return auditstore.AcceptRoundParams{}, nil, errors.New("next Round generated a non-finding item")
		}
		proposal, exists := descriptors[task.Finding.ReceiptID]
		if !exists || proposal.Digest != task.Finding.ProposalDigest {
			return auditstore.AcceptRoundParams{}, nil, errors.New("next Round proposal descriptor drifted")
		}
		itemID := auditdomain.DeterministicID(
			"item", snapshot.Audit.AuditID, fmt.Sprint(roundOrdinal), item.ItemKey,
		)
		approvalKind, approvalDigest, state, err := materializedItemApproval(
			snapshot.Audit.AuditID, profile, itemID, item, taskArtifacts[index],
		)
		if err != nil {
			return auditstore.AcceptRoundParams{}, nil, err
		}
		originRef := task.SourceRef
		items[index] = auditstore.MaterializedItem{
			ItemID:  itemID,
			ItemKey: item.ItemKey, Ordinal: item.Ordinal, Kind: item.Kind,
			SubjectKey: item.SubjectKey, Task: taskArtifacts[index], WorkflowRole: item.WorkflowRole,
			InitialState: state, ApprovalKind: approvalKind, ApprovalDigest: approvalDigest,
			Origin: auditstore.ItemOrigin{
				Schema: auditstore.ItemOriginSchema, SourceRef: &originRef,
				SourceContentDigest:      task.SourceContentDigest,
				SourceMediaType:          task.SourceMediaType,
				CanonicalInventoryDigest: task.CanonicalInventoryDigest,
				EntryKey:                 task.ItemKey,
			},
			Coverage: auditstore.Coverage{
				Status:    auditstore.CoverageNotTested,
				Requested: append([]string{}, inventory.Coverage.Rows[index].Requested...),
				Completed: []string{}, Gaps: append([]string{}, inventory.Coverage.Rows[index].Gaps...),
			},
			ProposalSources: []auditstore.ProposalItemSource{{
				ReceiptID:            task.Finding.ReceiptID,
				ProposedCheckOrdinal: task.Finding.ProposedCheckOrdinal,
				Proposal:             proposal,
			}},
		}
	}
	roundID := auditdomain.DeterministicID("round", snapshot.Audit.AuditID, fmt.Sprint(roundOrdinal))
	return auditstore.AcceptRoundParams{
		Claim: claim, ExpectedAuditRevision: snapshot.Audit.Revision,
		PreviousRoundID: snapshot.Round.RoundID, RoundID: roundID,
		RoundOrdinal: roundOrdinal, Manifest: worklist, Items: items,
	}, nil, nil
}
