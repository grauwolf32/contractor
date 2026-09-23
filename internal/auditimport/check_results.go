package auditimport

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type preparedCheckResults struct {
	members       []preparedMember
	evidenceByID  map[string]validatedEvidence
	evidenceOwner map[string]int
}

// invalidCheckResults distinguishes a stable contract rejection from an I/O
// failure that must propagate without changing the receipt disposition.
type invalidCheckResults struct{ code string }

func (e *invalidCheckResults) Error() string { return e.code }

func (i *Importer) collectSucceeded(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
	members []preparedMember,
	loadStandards retainedStandardLoader,
) (bool, error) {
	if execution.RunID == nil {
		return false, fmt.Errorf("%w: succeeded execution has no Run", ErrPermanent)
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(snapshot.Audit.ProfileSnapshot)
	if err != nil || profile.Ref.Name != snapshot.Audit.Profile.Name || profile.Ref.Version != snapshot.Audit.Profile.Version ||
		profile.Ref.Digest != snapshot.Audit.Profile.Digest {
		return false, fmt.Errorf("%w: pinned AuditProfile is invalid", ErrPermanent)
	}
	outputSlot, ok := resultOutputSlot(profile, members)
	if !ok {
		return i.collectTechnical(ctx, claim, execution, members,
			auditstore.CollectionMissingOutput, true, "result-output-contract-missing", auditstore.CoverageInconclusive)
	}
	source, payload, frozen, err := i.artifacts.ReadRunBinding(
		ctx, *execution.RunID, contracts.ArtifactRef{Namespace: "outputs", Name: outputSlot},
	)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		if isScanCheck(members) {
			return i.collectScanRecovery(ctx, claim, snapshot, execution, members, "scan_result_publication_missing")
		}
		return i.collectTechnical(ctx, claim, execution, members,
			auditstore.CollectionMissingOutput, true, "missing-output", auditstore.CoverageInconclusive)
	}
	if err != nil {
		return false, err
	}
	if source.MediaType != auditdomain.PackageMediaType || !frozen {
		if isScanCheck(members) {
			return i.collectScanRecovery(ctx, claim, snapshot, execution, members, "scan_result_publication_invalid")
		}
		return i.collectInvalid(ctx, claim, execution, members, source, "result-output-not-frozen-package")
	}
	manifestBytes, err := i.artifacts.ReadProjectExact(ctx, snapshot.Audit.ProjectID, execution.Manifest)
	if err != nil {
		return false, err
	}
	manifest, err := auditdomain.DecodeExecutionManifest(manifestBytes)
	if err != nil || auditdomain.ValidateDispatchExecutionManifest(manifest) != nil {
		return false, fmt.Errorf("%w: exact execution manifest is invalid", ErrPermanent)
	}
	resultPackage, err := auditdomain.DecodeCheckResultPackage(payload)
	if err != nil {
		if isScanCheck(members) {
			return i.collectScanRecovery(ctx, claim, snapshot, execution, members, "scan_result_package_invalid")
		}
		return i.collectInvalid(ctx, claim, execution, members, source, stableValidationCode(err))
	}
	if resultPackage.Package.Digest != source.Digest || auditdomain.ValidateResultSet(resultPackage.Results, manifest) != nil ||
		len(resultPackage.Results.Results) != len(members) {
		return i.collectInvalid(ctx, claim, execution, members, source, auditdomain.CodeResultSetInvalid)
	}

	prepared, err := i.prepareCheckResults(ctx, snapshot, execution, profile, resultPackage, members, loadStandards)
	if err != nil {
		var invalid *invalidCheckResults
		if errors.As(err, &invalid) {
			return i.collectInvalid(ctx, claim, execution, members, source, invalid.code)
		}
		return false, err
	}

	run, err := i.runs.GetRun(ctx, *execution.RunID)
	if err != nil {
		return false, err
	}
	if run.State != runstore.RunSucceeded || run.PublicationMode != runstore.PublicationAuditManaged ||
		run.ProjectID == nil || *run.ProjectID != snapshot.Audit.ProjectID ||
		run.AuditExecutionID == nil || *run.AuditExecutionID != execution.ExecutionID {
		return false, fmt.Errorf("%w: source Run identity is invalid", ErrPermanent)
	}
	if !retainedEvidenceFits(snapshot.Audit, source, prepared.evidenceByID) {
		return i.collectTechnical(ctx, claim, execution, members,
			auditstore.CollectionInvalidResult, false, "evidence-budget-exhausted",
			auditstore.CoverageInconclusive, &source)
	}
	return i.retainCheckResults(ctx, claim, snapshot, execution, run, source, prepared)
}

func (i *Importer) prepareCheckResults(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
	profile config.ResolvedAuditProfile,
	resultPackage auditdomain.CheckResultPackage,
	members []preparedMember,
	loadStandards retainedStandardLoader,
) (preparedCheckResults, error) {
	resultByKey := make(map[string]auditdomain.CheckResult, len(resultPackage.Results.Results))
	for _, value := range resultPackage.Results.Results {
		resultByKey[value.ItemKey] = value
	}
	evidenceByID := make(map[string]validatedEvidence, len(resultPackage.Evidence.Evidence))
	evidenceOwner := make(map[string]int, len(resultPackage.Evidence.Evidence))
	proposalOwner := make(map[string]int)
	for _, value := range resultPackage.Evidence.Evidence {
		validated := validatedEvidence{value: value}
		if value.Artifact != nil {
			descriptor, _, readErr := i.artifacts.ReadRunExact(ctx, *execution.RunID, *value.Artifact)
			if readErr != nil {
				if errors.Is(readErr, artifacts.ErrArtifactNotFound) ||
					errors.Is(readErr, artifacts.ErrExactRevisionRequired) ||
					errors.Is(readErr, artifacts.ErrInvalidName) {
					return preparedCheckResults{}, &invalidCheckResults{code: "evidence-reference-invalid"}
				}
				// Storage failures do not prove that the worker's reference is
				// invalid. Leave this execution collecting so the same terminal
				// result can be retried without consuming an item attempt.
				return preparedCheckResults{}, fmt.Errorf("read exact Audit evidence: %w", readErr)
			}
			validated.descriptor = &descriptor
		}
		evidenceByID[value.ID] = validated
	}
	for index := range members {
		value, exists := resultByKey[members[index].item.ItemKey]
		if !exists || value.SubjectKey != members[index].item.SubjectKey {
			return preparedCheckResults{}, &invalidCheckResults{code: auditdomain.CodeResultSetInvalid}
		}
		proposals, err := i.resolveMemberProposals(
			ctx, snapshot, execution, members[index], value.Proposals, proposalOwner, index, loadStandards,
		)
		if err != nil {
			return preparedCheckResults{}, err
		}
		members[index].proposals = proposals
		coverage, validationErr := semanticCoverage(profile.Mode, members[index].task, value, evidenceByID)
		if validationErr != nil {
			return preparedCheckResults{}, &invalidCheckResults{code: stableValidationCode(validationErr)}
		}
		members[index].result, members[index].cover = value, coverage
		for _, evidenceID := range value.EvidenceIDs {
			if _, duplicate := evidenceOwner[evidenceID]; duplicate {
				return preparedCheckResults{}, &invalidCheckResults{code: "evidence-item-membership-invalid"}
			}
			evidenceOwner[evidenceID] = index
		}
	}

	return preparedCheckResults{members: members, evidenceByID: evidenceByID, evidenceOwner: evidenceOwner}, nil
}

func (i *Importer) resolveMemberProposals(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
	member preparedMember,
	selections []auditdomain.ProposalSelection,
	proposalOwner map[string]int,
	index int,
	loadStandards retainedStandardLoader,
) ([]findingintake.ResolvedProposal, error) {
	if len(selections) == 0 && member.task.Finding == nil {
		return nil, nil
	}
	if i.findings == nil {
		return nil, &invalidCheckResults{code: auditdomain.CodeResultSetInvalid}
	}
	keys := make([]findingintake.ProposalKey, len(selections))
	for proposalIndex, proposal := range selections {
		keys[proposalIndex] = findingintake.ProposalKey{
			InvocationID: proposal.InvocationID, ClientKey: proposal.ClientKey,
		}
	}
	resolved := make([]findingintake.ResolvedProposal, 0, len(keys)+1)
	if len(keys) != 0 {
		selected, resolveErr := i.findings.ResolveAuditProposals(
			ctx, snapshot.Audit.OwnerID, snapshot.Audit.AuditID,
			execution.ExecutionID, *execution.RunID, keys,
		)
		if resolveErr != nil {
			if errors.Is(resolveErr, findingintake.ErrInvalid) ||
				errors.Is(resolveErr, findingintake.ErrNotFound) ||
				errors.Is(resolveErr, findingintake.ErrConflict) {
				return nil, &invalidCheckResults{code: "finding-proposal-association-invalid"}
			}
			return nil, resolveErr
		}
		for _, proposal := range selected {
			if owner, duplicate := proposalOwner[proposal.ReceiptID]; duplicate && owner != index {
				return nil, &invalidCheckResults{code: "finding-proposal-item-membership-invalid"}
			}
			proposalOwner[proposal.ReceiptID] = index
		}
		resolved = append(resolved, selected...)
	}
	if member.task.Finding != nil {
		proposal, resolveErr := i.resolveTaskProposal(ctx, snapshot, *member.task.Finding)
		if resolveErr != nil {
			if errors.Is(resolveErr, findingintake.ErrInvalid) ||
				errors.Is(resolveErr, findingintake.ErrNotFound) ||
				errors.Is(resolveErr, findingintake.ErrConflict) {
				return nil, &invalidCheckResults{code: "finding-task-proposal-invalid"}
			}
			return nil, resolveErr
		}
		resolved = append(resolved, proposal)
	}
	seenProposalReceipts := make(map[string]struct{}, len(resolved))
	for _, proposal := range resolved {
		if _, duplicate := seenProposalReceipts[proposal.ReceiptID]; duplicate {
			return nil, &invalidCheckResults{code: "finding-proposal-association-invalid"}
		}
		seenProposalReceipts[proposal.ReceiptID] = struct{}{}
		if validationErr := validateMappedProposalStandards(
			member.task, proposal.Document, loadStandards,
		); validationErr != nil {
			return nil, &invalidCheckResults{code: "finding-proposal-standard-reference-invalid"}
		}
	}
	return resolved, nil
}

func (i *Importer) retainCheckResults(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
	run runstore.WorkflowRun,
	source auditstore.ExactArtifact,
	prepared preparedCheckResults,
) (bool, error) {
	members, evidenceByID, evidenceOwner := prepared.members, prepared.evidenceByID, prepared.evidenceOwner
	namespace := auditdomain.ArtifactNamespace(snapshot.Audit.AuditID)
	retainedResult, err := i.artifacts.RetainRunExact(
		ctx, run.RunID, source, snapshot.Audit.ProjectID,
		contracts.ArtifactRef{Namespace: namespace, Name: auditdomain.DeterministicID("result", execution.ExecutionID)},
	)
	if err != nil {
		return false, err
	}
	links := make([]auditstore.ArtifactLink, 0, 1+len(evidenceByID))
	collectionItems := make([]auditstore.CollectionItem, len(members))
	for index, member := range members {
		provenance, provenanceErr := collectionProvenance(snapshot, execution, member, run, source)
		if provenanceErr != nil {
			return false, provenanceErr
		}
		links = append(links, auditstore.ArtifactLink{
			LogicalKey: "result/" + member.member.ExecutionItemID,
			Artifact:   retainedResult, SourceProvenance: provenance,
		})
		collectionItems[index] = auditstore.CollectionItem{
			ExecutionItemID: member.member.ExecutionItemID,
			Disposition:     auditstore.CollectionAccepted, Result: &retainedResult,
			FinalDisposition: auditstore.FinalAccepted, Coverage: member.cover,
		}
		for _, proposal := range member.proposals {
			collectionItems[index].FindingAssociations = append(
				collectionItems[index].FindingAssociations,
				auditstore.FindingAssociation{
					AssessmentID: auditdomain.DeterministicID(
						"finding-assessment", member.member.ExecutionItemID, proposal.ReceiptID,
					),
					ReceiptID: proposal.ReceiptID,
					Proposal: auditstore.ExactArtifact{
						Ref: proposal.Proposal.Ref, Digest: proposal.Proposal.Digest,
						MediaType: proposal.Proposal.MediaType, SizeBytes: proposal.Proposal.SizeBytes,
					},
					SemanticAssessment: member.result.Assessment,
				},
			)
		}
	}
	// Copy each distinct exact source once. Evidence records citing the same
	// revision, or the result output itself, share that copy, so the store
	// charges exactly the bytes retainedEvidenceFits admitted. The first
	// sorted evidence ID names a copy, keeping retries deterministic.
	retainedBySource := map[string]auditstore.ExactArtifact{exactRefKey(source.Ref): retainedResult}
	evidenceIDs := sortedEvidenceIDs(evidenceByID)
	for _, id := range evidenceIDs {
		evidence := evidenceByID[id]
		artifact := retainedResult
		displayRef := "member:" + evidence.value.ContentMemberID
		if evidence.descriptor != nil {
			key := exactRefKey(evidence.descriptor.Ref)
			shared, retained := retainedBySource[key]
			if !retained {
				shared, err = i.artifacts.RetainRunExact(
					ctx, run.RunID, *evidence.descriptor, snapshot.Audit.ProjectID,
					contracts.ArtifactRef{Namespace: namespace, Name: auditdomain.DeterministicID("evidence", execution.ExecutionID, id)},
				)
				if err != nil {
					return false, err
				}
				retainedBySource[key] = shared
			}
			artifact, displayRef = shared, ""
		}
		owner, exists := evidenceOwner[id]
		if !exists || owner < 0 || owner >= len(members) {
			return i.collectInvalid(ctx, claim, execution, members, source, "evidence-item-membership-invalid")
		}
		provenance, provenanceErr := evidenceProvenance(snapshot, execution, members[owner], run, source, evidence.value)
		if provenanceErr != nil {
			return false, provenanceErr
		}
		links = append(links, auditstore.ArtifactLink{
			LogicalKey: "evidence/" + execution.ExecutionID + "/" + id,
			Artifact:   artifact, SourceProvenance: provenance, DisplayRef: displayRef,
		})
	}
	changed, err := i.commitCollection(ctx, claim, execution, auditstore.CollectionAccepted,
		&source, links, nil, collectionItems)
	if errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
		// The snapshot budget admitted these bytes, but a concurrent writer
		// consumed it first. Settle exactly as the pre-check would have; the
		// unlinked copies stay in the Audit namespace until Audit purge.
		return i.collectTechnical(ctx, claim, execution, members,
			auditstore.CollectionInvalidResult, false, "evidence-budget-exhausted",
			auditstore.CoverageInconclusive, &source)
	}
	return changed, err
}
