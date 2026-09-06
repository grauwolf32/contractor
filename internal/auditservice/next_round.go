package auditservice

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
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
	selected, eligible, scanExhausted, err := s.selectNextProposalChecks(
		ctx, snapshot.Audit, capacity,
	)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	if len(selected) == 0 {
		switch {
		case eligible && snapshot.Round.Ordinal >= snapshot.Audit.Limits.MaxRounds:
			return auditstore.AcceptRoundParams{}, &auditstore.StopReason{
				Code:    "round_budget_exhausted",
				Message: "Unscheduled finding checks remain after the configured Round budget was exhausted.",
			}, nil
		case eligible && capacity <= 0:
			return auditstore.AcceptRoundParams{}, &auditstore.StopReason{
				Code:    "item_budget_exhausted",
				Message: "Unscheduled finding checks remain after the configured Audit item budget was exhausted.",
			}, nil
		case scanExhausted:
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
			Name:      deterministicID("proposal-inventory", snapshot.Audit.AuditID, digestBytes(sourceBytes)),
		},
		artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: sourceBytes},
	)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	approval := auditdomain.ApprovalNone
	if profile.Interaction.ActiveChecks == config.AuditActiveChecksApprovalRequired {
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
		state := auditstore.ItemReady
		if item.ApprovalRequirement != auditdomain.ApprovalNone {
			state = auditstore.ItemAwaitingReview
		}
		originRef := task.SourceRef
		items[index] = auditstore.MaterializedItem{
			ItemID: deterministicID(
				"item", snapshot.Audit.AuditID, fmt.Sprint(roundOrdinal), item.ItemKey,
			),
			ItemKey: item.ItemKey, Ordinal: item.Ordinal, Kind: item.Kind,
			SubjectKey: item.SubjectKey, Task: taskArtifacts[index], WorkflowRole: item.WorkflowRole,
			InitialState: state,
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
	roundID := deterministicID("round", snapshot.Audit.AuditID, fmt.Sprint(roundOrdinal))
	return auditstore.AcceptRoundParams{
		Claim: claim, ExpectedAuditRevision: snapshot.Audit.Revision,
		PreviousRoundID: snapshot.Round.RoundID, RoundID: roundID,
		RoundOrdinal: roundOrdinal, Manifest: worklist, Items: items,
	}, nil, nil
}

func (s *Service) selectNextProposalChecks(
	ctx context.Context,
	audit auditstore.Audit,
	capacity int,
) ([]auditdomain.FindingInventoryProposal, bool, bool, error) {
	selected := make(map[string]auditdomain.FindingInventoryProposal)
	selectedCount := 0
	eligible := false
	scanned := 0
	query := findingintake.ListQuery{Limit: nextRoundInboxPage}
	for scanned < maxNextRoundInboxScan {
		receipts, err := s.findings.ListAuditInbox(ctx, audit.OwnerID, audit.AuditID, query)
		if err != nil {
			return nil, false, false, err
		}
		ids := make([]string, len(receipts))
		for index := range receipts {
			ids[index] = receipts[index].ReceiptID
		}
		used, err := s.scheduledProposalChecks(ctx, audit.AuditID, ids)
		if err != nil {
			return nil, false, false, err
		}
		for _, receipt := range receipts {
			scanned++
			hold, found := exactAuditHold(receipt, audit.AuditID, audit.ProjectID)
			if !found {
				return nil, false, false, fmt.Errorf("Audit inbox receipt %q has no exact hold", receipt.ReceiptID)
			}
			for ordinal := range receipt.Document.ProposedChecks {
				if used[auditdomain.ReceiptCheckIdentity(receipt.ReceiptID, ordinal)] {
					continue
				}
				eligible = true
				if capacity <= 0 || selectedCount >= capacity {
					continue
				}
				candidate, exists := selected[receipt.ReceiptID]
				if !exists {
					candidate = auditdomain.FindingInventoryProposal{
						ReceiptID: receipt.ReceiptID,
						Proposal: auditdomain.FindingInventoryArtifact{
							Ref: hold.Proposal.Ref, Digest: hold.Proposal.Digest,
							MediaType: hold.Proposal.MediaType, SizeBytes: hold.Proposal.SizeBytes,
						},
						Document: receipt.Document, SelectedCheckOrdinals: []int{},
					}
				}
				candidate.SelectedCheckOrdinals = append(candidate.SelectedCheckOrdinals, ordinal)
				selected[receipt.ReceiptID] = candidate
				selectedCount++
			}
			if scanned >= maxNextRoundInboxScan {
				break
			}
		}
		if len(receipts) < query.Limit || (selectedCount >= capacity && capacity > 0) ||
			(eligible && capacity <= 0) {
			break
		}
		last := receipts[len(receipts)-1]
		query.AfterCreatedAt, query.AfterReceiptID = &last.CreatedAt, last.ReceiptID
	}
	result := make([]auditdomain.FindingInventoryProposal, 0, len(selected))
	for _, proposal := range selected {
		result = append(result, proposal)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ReceiptID < result[j].ReceiptID })
	return result, eligible, scanned >= maxNextRoundInboxScan, nil
}

func (s *Service) scheduledProposalChecks(
	ctx context.Context, auditID string, receiptIDs []string,
) (map[string]bool, error) {
	result := make(map[string]bool)
	if len(receiptIDs) == 0 {
		return result, nil
	}
	rows, err := s.pool.Query(ctx, `
SELECT receipt_id, proposed_check_ordinal
  FROM audit_proposal_items
 WHERE audit_id = $1 AND receipt_id = ANY($2::text[])`, auditID, receiptIDs)
	if err != nil {
		return nil, fmt.Errorf("list scheduled Audit proposal checks: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var receiptID string
		var ordinal int
		if err := rows.Scan(&receiptID, &ordinal); err != nil {
			return nil, err
		}
		result[auditdomain.ReceiptCheckIdentity(receiptID, ordinal)] = true
	}
	return result, rows.Err()
}

func exactAuditHold(receipt findingintake.Receipt, auditID, projectID string) (findingintake.AuditHold, bool) {
	for _, hold := range receipt.AuditHolds {
		if hold.AuditID == auditID && hold.ProjectID == projectID {
			return hold, true
		}
	}
	return findingintake.AuditHold{}, false
}
