package auditimport

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
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
	profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(snapshot.Audit.ProfileSnapshot)
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
	profile auditprofile.ResolvedAuditProfile,
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
	descriptors, err := i.describeEvidence(ctx, *execution.RunID, resultPackage.Evidence.Evidence)
	if err != nil {
		return preparedCheckResults{}, err
	}
	for _, value := range resultPackage.Evidence.Evidence {
		validated := validatedEvidence{value: value}
		if value.Artifact != nil {
			descriptor := descriptors[value.Artifact.Key()]
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
	// Every evidence record now has exactly one owning member index.
	// DecodeCheckResultPackage rejects an evidence record no result cites;
	// collectSucceeded admits exactly len(members) results with distinct
	// item keys; the members of one execution are distinct Items of one Round
	// (audit_execution_items FKs, audit_items_round_key), so the loop above
	// matched every result to one member and recorded each cited ID once.
	return preparedCheckResults{members: members, evidenceByID: evidenceByID, evidenceOwner: evidenceOwner}, nil
}

// describeEvidence returns the exact descriptor of every distinct revision
// that evidence cites, keyed by reference, from metadata reads that fetch no
// payload: digest, media type and size are all that the evidence budget,
// retention and the receipt need, and the retained binding shares the stored
// version whose bytes every reader verifies (AUX-R98). A reference that is not
// an exact revision of the Run, or that the Run does not hold, is
// evidence-reference-invalid. Any other failure does not prove the Worker's
// reference invalid, so it leaves this execution collecting and the same
// terminal result is retried without consuming an item attempt (AUX-R96).
func (i *Importer) describeEvidence(
	ctx context.Context, runID string, evidence []auditdomain.Evidence,
) (map[string]auditstore.ExactArtifact, error) {
	refs := make([]contracts.ArtifactRef, 0, len(evidence))
	seen := make(map[string]struct{}, len(evidence))
	for _, value := range evidence {
		if value.Artifact == nil {
			continue
		}
		if _, duplicate := seen[value.Artifact.Key()]; duplicate {
			continue
		}
		seen[value.Artifact.Key()] = struct{}{}
		refs = append(refs, *value.Artifact)
	}
	if len(refs) == 0 {
		return nil, nil
	}
	described, err := i.artifacts.DescribeRunExact(ctx, runID, refs)
	if errors.Is(err, artifacts.ErrArtifactNotFound) ||
		errors.Is(err, artifacts.ErrExactRevisionRequired) ||
		errors.Is(err, artifacts.ErrInvalidName) {
		return nil, &invalidCheckResults{code: "evidence-reference-invalid"}
	}
	if err != nil {
		return nil, fmt.Errorf("describe exact Audit evidence: %w", err)
	}
	if len(described) != len(refs) {
		return nil, fmt.Errorf("describe exact Audit evidence: %d descriptors for %d references", len(described), len(refs))
	}
	descriptors := make(map[string]auditstore.ExactArtifact, len(refs))
	for index, ref := range refs {
		descriptors[ref.Key()] = described[index]
	}
	return descriptors, nil
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
		if err := validateResolvedProposalStandards(member.task, proposal.Document, loadStandards); err != nil {
			return nil, err
		}
	}
	return resolved, nil
}

// validateResolvedProposalStandards separates the pinned standards from the
// Worker's proposal. Loading them keeps its class: a storage failure stays
// retryable and invalid pinned standards are ErrPermanent, also when every
// proposal was retained by an earlier attempt that skipped this read. Only a
// proposal whose standard references violate their contract is invalid-result.
func validateResolvedProposalStandards(
	task auditdomain.ItemTask,
	document auditdomain.FindingProposal,
	loadStandards retainedStandardLoader,
) error {
	if task.Standard == nil && len(document.StandardRefs) == 0 {
		return nil
	}
	standards, err := loadStandards()
	if err != nil {
		return err
	}
	if task.Standard != nil {
		standard, exists := standards[retainedStandardKey(task.Standard.Scheme, task.Standard.Version)]
		if !exists || !standardTaskMatchesPackage(task, standard) {
			return fmt.Errorf("%w: pinned standard task does not match its retained package", ErrPermanent)
		}
	}
	if validateMappedProposalStandards(task, document, standards) != nil {
		return &invalidCheckResults{code: "finding-proposal-standard-reference-invalid"}
	}
	return nil
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
	// Copy each distinct exact source once. Evidence records citing the same
	// revision, or the result output itself, share that copy, so the store
	// charges exactly the bytes retainedEvidenceFits admitted. The first
	// sorted evidence ID names a copy, keeping retries deterministic. All
	// copies are retained in bounded artifact-write transactions (AUX-R98)
	// before the one collection transaction links them.
	retentions := []RunRetention{{
		Source: source,
		Target: contracts.ArtifactRef{Namespace: namespace, Name: auditdomain.DeterministicID("result", execution.ExecutionID)},
	}}
	retentionBySource := map[string]int{source.Ref.Key(): 0}
	evidenceIDs := sortedEvidenceIDs(evidenceByID)
	for _, id := range evidenceIDs {
		descriptor := evidenceByID[id].descriptor
		if descriptor == nil {
			continue
		}
		if _, planned := retentionBySource[descriptor.Ref.Key()]; planned {
			continue
		}
		retentionBySource[descriptor.Ref.Key()] = len(retentions)
		retentions = append(retentions, RunRetention{
			Source: *descriptor,
			Target: contracts.ArtifactRef{Namespace: namespace, Name: auditdomain.DeterministicID("evidence", execution.ExecutionID, id)},
		})
	}
	retained, err := i.artifacts.RetainRunExact(ctx, run.RunID, snapshot.Audit.ProjectID, retentions)
	if err != nil {
		return false, err
	}
	if len(retained) != len(retentions) {
		return false, fmt.Errorf("retain Audit collection material: %d of %d revisions", len(retained), len(retentions))
	}
	retainedResult := retained[0]
	links := make([]auditstore.ArtifactLink, 0, 1+len(evidenceByID))
	collectionItems := make([]auditstore.CollectionItem, len(members))
	for index, member := range members {
		provenance, provenanceErr := collectionProvenance(snapshot, execution, member, run, source)
		if provenanceErr != nil {
			return false, provenanceErr
		}
		links = append(links, auditstore.ArtifactLink{Scope: "project", LogicalKey: "result/" + member.member.ExecutionItemID,
			Artifact: retainedResult, SourceProvenance: provenance,
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
	for _, id := range evidenceIDs {
		evidence := evidenceByID[id]
		artifact := retainedResult
		displayRef := "member:" + evidence.value.ContentMemberID
		if evidence.descriptor != nil {
			artifact, displayRef = retained[retentionBySource[evidence.descriptor.Ref.Key()]], ""
		}
		// prepareCheckResults established one owning member for every ID.
		provenance, provenanceErr := evidenceProvenance(snapshot, execution, members[evidenceOwner[id]], run, source, evidence.value)
		if provenanceErr != nil {
			return false, provenanceErr
		}
		links = append(links, auditstore.ArtifactLink{Scope: "project", LogicalKey: "evidence/" + execution.ExecutionID + "/" + id,
			Artifact: artifact, SourceProvenance: provenance, DisplayRef: displayRef,
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
