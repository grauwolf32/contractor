package auditimport

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const provenanceSchema = "contractor.audit.collection-provenance.v1"

type preparedMember struct {
	member auditstore.ExecutionItem
	item   auditstore.Item
	task   auditdomain.ItemTask
	result auditdomain.CheckResult
	cover  auditstore.Coverage
	origin json.RawMessage
}

type validatedEvidence struct {
	value      auditdomain.Evidence
	descriptor *auditstore.ExactArtifact
}

// Collect converts one exact terminal observation into one durable receipt.
// It never changes the child WorkflowRun outcome.
func (i *Importer) Collect(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
) (bool, error) {
	changed, err := i.collect(ctx, claim, snapshot, execution)
	if !errors.Is(err, ErrPermanent) {
		return changed, err
	}
	// A terminal Run must always become disposable. If exact pinned data is
	// corrupt, retain a truthful technical receipt rather than retrying an
	// impossible successful import forever. The ordinary Collect transaction
	// revalidates authority and the terminal observation.
	if execution.AuditID != snapshot.Audit.AuditID || execution.AuditID != claim.AuditID ||
		execution.State != auditstore.ExecutionCollecting || execution.TerminalOutcome == nil {
		return false, err
	}
	return i.collectContractInvalid(ctx, claim, execution)
}

func (i *Importer) collect(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
) (bool, error) {
	if execution.AuditID != snapshot.Audit.AuditID || execution.AuditID != claim.AuditID ||
		execution.State != auditstore.ExecutionCollecting || execution.TerminalOutcome == nil {
		return false, fmt.Errorf("%w: collection snapshot is inconsistent", ErrPermanent)
	}
	members, err := i.store.ListExecutionItems(ctx, execution.ExecutionID)
	if err != nil {
		return false, err
	}
	if len(members) == 0 || len(members) > auditstore.MaxCollectionItems {
		return false, fmt.Errorf("%w: collection membership is invalid", ErrPermanent)
	}
	prepared, err := i.prepareMembers(ctx, snapshot, members)
	if err != nil {
		return false, err
	}
	if err := i.retainFindingProposals(ctx, snapshot, execution); err != nil {
		return false, err
	}

	switch *execution.TerminalOutcome {
	case auditstore.TerminalFailed, auditstore.TerminalSubmissionFailed:
		return i.collectTechnical(ctx, claim, execution, prepared,
			auditstore.CollectionExecutionFailed, true, "execution-failed", auditstore.CoverageBlocked)
	case auditstore.TerminalCancelled:
		return i.collectTechnical(ctx, claim, execution, prepared,
			auditstore.CollectionExecutionCancelled, false, "execution-cancelled", auditstore.CoverageBlocked)
	case auditstore.TerminalSucceeded:
		return i.collectSucceeded(ctx, claim, snapshot, execution, prepared)
	default:
		return false, fmt.Errorf("%w: terminal outcome is unsupported", ErrPermanent)
	}
}

// retainFindingProposals runs before the collection receipt commits. The
// transfer is independently idempotent, so a later collection retry resumes
// safely. A Run can fail after committing a proposal; technical failure must
// not erase that candidate or silently promote it to a confirmed finding.
func (i *Importer) retainFindingProposals(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
) error {
	if i.findings == nil || execution.RunID == nil {
		return nil
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(snapshot.Audit.ProfileSnapshot)
	if err != nil {
		return fmt.Errorf("%w: pinned AuditProfile is invalid", ErrPermanent)
	}
	query := findingintake.ListQuery{Limit: 200}
	for {
		receipts, err := i.findings.ListRun(
			ctx, snapshot.Audit.OwnerID, *execution.RunID, query,
		)
		if err != nil {
			return fmt.Errorf("list Audit child finding proposals: %w", err)
		}
		if profile.Interaction.FindingConfirmation == config.AuditFindingDisabled && len(receipts) != 0 {
			return fmt.Errorf("%w: findings-disabled Audit child produced a proposal", ErrPermanent)
		}
		for _, receipt := range receipts {
			if receipt.Origin.Audit == nil || receipt.Origin.Audit.AuditID != snapshot.Audit.AuditID ||
				receipt.Origin.Audit.ExecutionID != execution.ExecutionID ||
				receipt.Origin.RunID != *execution.RunID {
				return fmt.Errorf("%w: Audit child finding origin is inconsistent", ErrPermanent)
			}
			if _, _, err := i.findings.ImportIntoAudit(ctx, findingintake.ImportRequest{
				OwnerID: snapshot.Audit.OwnerID, AuditID: snapshot.Audit.AuditID,
				RunID: *execution.RunID, Proposal: receipt.Proposal.Ref,
			}); err != nil {
				return fmt.Errorf("retain Audit child finding proposal: %w", err)
			}
		}
		if len(receipts) < query.Limit {
			return nil
		}
		last := receipts[len(receipts)-1]
		query.AfterCreatedAt = &last.CreatedAt
		query.AfterReceiptID = last.ReceiptID
	}
}

func (i *Importer) collectContractInvalid(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	execution auditstore.Execution,
) (bool, error) {
	members, err := i.store.ListExecutionItems(ctx, execution.ExecutionID)
	if err != nil {
		return false, err
	}
	if len(members) > auditstore.MaxCollectionItems {
		return false, fmt.Errorf("%w: collection membership exceeds the durable receipt bound", ErrPermanent)
	}
	const code = "collection-contract-invalid"
	items := make([]auditstore.CollectionItem, len(members))
	for index, member := range members {
		items[index] = auditstore.CollectionItem{
			ExecutionItemID:  member.ExecutionItemID,
			Disposition:      auditstore.CollectionContractInvalid,
			FinalDisposition: auditstore.FinalInvalidResult,
			Coverage: auditstore.Coverage{
				Status: auditstore.CoverageBlocked, Requested: []string{}, Completed: []string{},
				Gaps: []string{code}, Rationale: "Trusted collection could not validate its pinned contract.",
			},
		}
	}
	errorCode := code
	return i.commitCollection(
		ctx, claim, execution, auditstore.CollectionContractInvalid, nil, nil, &errorCode, items,
	)
}

func (i *Importer) prepareMembers(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	members []auditstore.ExecutionItem,
) ([]preparedMember, error) {
	result := make([]preparedMember, len(members))
	for index, member := range members {
		item, ok := findItem(snapshot.Items, member.ItemID)
		if !ok || item.State != auditstore.ItemCollecting || member.State != auditstore.ItemCollecting ||
			item.RoundID != member.RoundID || item.Task.Digest != member.Task.Digest || !sameRef(item.Task.Ref, member.Task.Ref) {
			return nil, fmt.Errorf("%w: collecting item membership is inconsistent", ErrPermanent)
		}
		payload, err := i.artifacts.ReadProjectExact(ctx, snapshot.Audit.ProjectID, member.Task)
		if err != nil {
			return nil, fmt.Errorf("read exact Audit task package: %w", err)
		}
		task, err := decodeTaskPackage(payload, member, item)
		if err != nil {
			return nil, fmt.Errorf("%w: exact Audit task package is invalid", ErrPermanent)
		}
		origin, err := encodeTaskOrigin(task)
		if err != nil {
			return nil, err
		}
		result[index] = preparedMember{
			member: member, item: item, task: task, origin: origin,
			cover: baselineCoverage(task),
		}
	}
	return result, nil
}

func (i *Importer) collectSucceeded(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
	members []preparedMember,
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
		return i.collectTechnical(ctx, claim, execution, members,
			auditstore.CollectionMissingOutput, true, "missing-output", auditstore.CoverageInconclusive)
	}
	if err != nil {
		return false, err
	}
	if source.MediaType != auditdomain.PackageMediaType || !frozen {
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
		return i.collectInvalid(ctx, claim, execution, members, source, stableValidationCode(err))
	}
	if resultPackage.Package.Digest != source.Digest || auditdomain.ValidateResultSet(resultPackage.Results, manifest) != nil ||
		len(resultPackage.Results.Results) != len(members) {
		return i.collectInvalid(ctx, claim, execution, members, source, auditdomain.CodeResultSetInvalid)
	}

	resultByKey := make(map[string]auditdomain.CheckResult, len(resultPackage.Results.Results))
	for _, value := range resultPackage.Results.Results {
		resultByKey[value.ItemKey] = value
	}
	evidenceByID := make(map[string]validatedEvidence, len(resultPackage.Evidence.Evidence))
	for _, value := range resultPackage.Evidence.Evidence {
		validated := validatedEvidence{value: value}
		if value.Artifact != nil {
			descriptor, _, readErr := i.artifacts.ReadRunExact(ctx, *execution.RunID, *value.Artifact)
			if readErr != nil {
				return i.collectInvalid(ctx, claim, execution, members, source, "evidence-reference-invalid")
			}
			validated.descriptor = &descriptor
		}
		evidenceByID[value.ID] = validated
	}
	for index := range members {
		value, exists := resultByKey[members[index].item.ItemKey]
		if !exists || value.SubjectKey != members[index].item.SubjectKey || len(value.Proposals) != 0 {
			return i.collectInvalid(ctx, claim, execution, members, source, auditdomain.CodeResultSetInvalid)
		}
		coverage, validationErr := semanticCoverage(profile.Mode, members[index].task, value, evidenceByID)
		if validationErr != nil {
			return i.collectInvalid(ctx, claim, execution, members, source, stableValidationCode(validationErr))
		}
		members[index].result, members[index].cover = value, coverage
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
	if !retainedEvidenceFits(snapshot.Audit, source, evidenceByID) {
		return i.collectTechnical(ctx, claim, execution, members,
			auditstore.CollectionInvalidResult, false, "evidence-budget-exhausted",
			auditstore.CoverageInconclusive, &source)
	}
	namespace := auditdomain.ArtifactNamespace(snapshot.Audit.AuditID)
	retainedResult, err := i.artifacts.RetainRunExact(
		ctx, run.RunID, source, snapshot.Audit.ProjectID,
		contracts.ArtifactRef{Namespace: namespace, Name: deterministicID("result", execution.ExecutionID)},
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
	}
	evidenceIDs := sortedEvidenceIDs(evidenceByID)
	for _, id := range evidenceIDs {
		evidence := evidenceByID[id]
		artifact := retainedResult
		displayRef := "member:" + evidence.value.ContentMemberID
		if evidence.descriptor != nil {
			artifact, err = i.artifacts.RetainRunExact(
				ctx, run.RunID, *evidence.descriptor, snapshot.Audit.ProjectID,
				contracts.ArtifactRef{Namespace: namespace, Name: deterministicID("evidence", execution.ExecutionID, id)},
			)
			if err != nil {
				return false, err
			}
			displayRef = ""
		}
		provenance, provenanceErr := evidenceProvenance(snapshot, execution, members[0], run, source, evidence.value)
		if provenanceErr != nil {
			return false, provenanceErr
		}
		links = append(links, auditstore.ArtifactLink{
			LogicalKey: "evidence/" + execution.ExecutionID + "/" + id,
			Artifact:   artifact, SourceProvenance: provenance, DisplayRef: displayRef,
		})
	}
	return i.commitCollection(ctx, claim, execution, auditstore.CollectionAccepted,
		&source, links, nil, collectionItems)
}

func (i *Importer) collectInvalid(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	execution auditstore.Execution,
	members []preparedMember,
	source auditstore.ExactArtifact,
	code string,
) (bool, error) {
	return i.collectTechnical(ctx, claim, execution, members,
		auditstore.CollectionInvalidResult, true, code, auditstore.CoverageInconclusive, &source)
}

func (i *Importer) collectTechnical(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	execution auditstore.Execution,
	members []preparedMember,
	disposition auditstore.CollectionDisposition,
	retryable bool,
	code string,
	status auditstore.CoverageStatus,
	optionalSource ...*auditstore.ExactArtifact,
) (bool, error) {
	items := make([]auditstore.CollectionItem, len(members))
	for index, member := range members {
		coverage := member.cover
		coverage.Status = status
		coverage.Gaps = mergeSorted(coverage.Gaps, []string{code})
		coverage.Rationale = technicalRationale(code)
		items[index] = auditstore.CollectionItem{
			ExecutionItemID: member.member.ExecutionItemID,
			Disposition:     disposition, Retryable: retryable,
			FinalDisposition: finalDisposition(disposition), Coverage: coverage,
		}
	}
	var source *auditstore.ExactArtifact
	if len(optionalSource) != 0 {
		source = optionalSource[0]
	}
	return i.commitCollection(ctx, claim, execution, disposition, source, nil, &code, items)
}

func (i *Importer) commitCollection(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	execution auditstore.Execution,
	disposition auditstore.CollectionDisposition,
	source *auditstore.ExactArtifact,
	retained []auditstore.ArtifactLink,
	errorCode *string,
	items []auditstore.CollectionItem,
) (bool, error) {
	identity := struct {
		Schema      string                           `json:"schema"`
		ExecutionID string                           `json:"executionId"`
		Outcome     *auditstore.TerminalOutcome      `json:"outcome"`
		Generation  *string                          `json:"generation"`
		Sequence    *uint64                          `json:"sequence"`
		Disposition auditstore.CollectionDisposition `json:"disposition"`
		Source      *auditstore.ExactArtifact        `json:"source,omitempty"`
		Retained    []auditstore.ArtifactLink        `json:"retained"`
		ErrorCode   *string                          `json:"errorCode,omitempty"`
		Items       []auditstore.CollectionItem      `json:"items"`
	}{
		Schema: "contractor.audit.collection-request.v1", ExecutionID: execution.ExecutionID,
		Outcome: execution.TerminalOutcome, Generation: execution.TerminalRunGeneration,
		Sequence: execution.TerminalRunSequence, Disposition: disposition,
		Source: source, Retained: nonNilLinks(retained), ErrorCode: errorCode, Items: items,
	}
	encoded, err := json.Marshal(identity)
	if err != nil {
		return false, err
	}
	_, _, err = i.store.Collect(ctx, auditstore.CollectParams{
		Claim: claim, ReceiptID: deterministicID("receipt", execution.ExecutionID),
		ExecutionID: execution.ExecutionID, Disposition: disposition,
		SourceOutput: source, Retained: nonNilLinks(retained), ErrorCode: errorCode,
		RequestDigest: digestBytes(encoded), Items: items,
	})
	return err == nil, err
}

func resultOutputSlot(profile config.ResolvedAuditProfile, members []preparedMember) (string, bool) {
	if len(members) == 0 {
		return "", false
	}
	role := members[0].item.WorkflowRole
	binding, exists := profile.Workflows[role]
	if !exists {
		return "", false
	}
	for _, member := range members[1:] {
		if member.item.WorkflowRole != role {
			return "", false
		}
	}
	output, exists := binding.Outputs["result"]
	if !exists || output == "" {
		return "", false
	}
	contract, exists := binding.Workflow.Outputs[output]
	if !exists || !contract.Required || !contains(contract.MediaTypes, auditdomain.PackageMediaType) {
		return "", false
	}
	return output, true
}

func decodeTaskPackage(
	payload []byte, member auditstore.ExecutionItem, item auditstore.Item,
) (auditdomain.ItemTask, error) {
	pkg, err := auditdomain.ValidatePackage(payload)
	if err != nil || pkg.Digest != member.Task.Digest || pkg.Manifest.Kind != auditdomain.PackageKindTask || len(pkg.Members()) != 1 {
		return auditdomain.ItemTask{}, auditdomainError()
	}
	document, exists := pkg.MemberByID("task-document")
	if !exists || document.Metadata().Path != "task.json" || document.Metadata().MediaType != auditdomain.JSONMediaType {
		return auditdomain.ItemTask{}, auditdomainError()
	}
	task, err := auditdomain.DecodeItemTask(document.Data())
	if err != nil || task.ItemKey != item.ItemKey || task.SubjectKey != item.SubjectKey ||
		task.WorkflowRole != item.WorkflowRole || task.Kind != item.Kind {
		return auditdomain.ItemTask{}, auditdomainError()
	}
	return task, nil
}

func auditdomainError() error {
	return fmt.Errorf("%s", auditdomain.CodeInventoryInvalid)
}

func semanticCoverage(
	mode config.AuditProfileMode,
	task auditdomain.ItemTask,
	result auditdomain.CheckResult,
	evidence map[string]validatedEvidence,
) (auditstore.Coverage, error) {
	expected := expectedCoverage(task)
	requested := sortedCopy(result.Coverage.Requested)
	if !equalStrings(expected, requested) {
		return auditstore.Coverage{}, fmt.Errorf("%s", auditdomain.CodeResultSetInvalid)
	}
	completed := sortedCopy(result.Coverage.Completed)
	gaps := mergeSorted(taskGaps(task), result.Coverage.Gaps)
	coverage := auditstore.Coverage{Requested: requested, Completed: completed, Gaps: gaps}

	if task.Operation != nil {
		if result.Assessment == "not-tested" {
			if len(completed) != 0 {
				return auditstore.Coverage{}, fmt.Errorf("%s", auditdomain.CodeResultSetInvalid)
			}
			coverage.Status = auditstore.CoverageNotTested
		} else if len(completed) == len(requested) && len(gaps) == 0 {
			coverage.Status = auditstore.CoverageTracedComplete
		} else if len(completed) != 0 {
			coverage.Status = auditstore.CoverageTracedPartial
		} else if len(gaps) != 0 {
			coverage.Status = auditstore.CoverageUnmapped
		} else {
			coverage.Status = auditstore.CoverageNotTested
		}
		return coverage, nil
	}

	if result.Assessment == "not-applicable" {
		return auditstore.Coverage{}, fmt.Errorf("%s", auditdomain.CodeResultSetInvalid)
	}
	conclusive := result.Assessment == "satisfied" || result.Assessment == "violated" ||
		result.Assessment == "supported" || result.Assessment == "refuted"
	if conclusive && !requiredEvidencePresent(task, result, evidence) {
		return auditstore.Coverage{}, fmt.Errorf("%s", auditdomain.CodeResultSetInvalid)
	}
	switch result.Assessment {
	case "satisfied":
		if len(completed) != len(requested) || len(gaps) != 0 {
			coverage.Status = auditstore.CoverageInconclusive
		} else {
			coverage.Status = auditstore.CoverageSatisfied
		}
	case "violated":
		if len(completed) != len(requested) || len(gaps) != 0 {
			coverage.Status = auditstore.CoverageInconclusive
		} else {
			coverage.Status = auditstore.CoverageViolated
		}
	case "supported":
		if mode != config.AuditModeRiskAssessment {
			return auditstore.Coverage{}, fmt.Errorf("%s", auditdomain.CodeResultSetInvalid)
		}
		if len(completed) != len(requested) || len(gaps) != 0 {
			coverage.Status = auditstore.CoverageInconclusive
		} else {
			coverage.Status = auditstore.CoverageViolated
		}
	case "refuted":
		if mode != config.AuditModeRiskAssessment || len(completed) != len(requested) || len(gaps) != 0 {
			coverage.Status = auditstore.CoverageInconclusive
		} else {
			coverage.Status = auditstore.CoverageSatisfied
		}
	case "blocked":
		coverage.Status = auditstore.CoverageBlocked
	case "inconclusive":
		coverage.Status = auditstore.CoverageInconclusive
	case "not-tested":
		coverage.Status = auditstore.CoverageNotTested
	default:
		return auditstore.Coverage{}, fmt.Errorf("%s", auditdomain.CodeResultSetInvalid)
	}
	return coverage, nil
}

func requiredEvidencePresent(
	task auditdomain.ItemTask,
	result auditdomain.CheckResult,
	evidence map[string]validatedEvidence,
) bool {
	if task.Checklist == nil {
		return true
	}
	kinds := make(map[string]struct{}, len(result.EvidenceIDs))
	for _, id := range result.EvidenceIDs {
		value, exists := evidence[id]
		if !exists {
			return false
		}
		kinds[value.value.Kind] = struct{}{}
	}
	for _, required := range task.Checklist.RequiredEvidence {
		if _, exists := kinds[required]; !exists {
			return false
		}
	}
	return true
}

func baselineCoverage(task auditdomain.ItemTask) auditstore.Coverage {
	return auditstore.Coverage{
		Status: auditstore.CoverageNotTested, Requested: expectedCoverage(task),
		Completed: []string{}, Gaps: taskGaps(task),
	}
}

func expectedCoverage(task auditdomain.ItemTask) []string {
	if task.Checklist != nil {
		return sortedCopy(task.Checklist.RequiredEvidence)
	}
	return []string{"operation-resolution"}
}

func taskGaps(task auditdomain.ItemTask) []string {
	if task.Operation == nil {
		return []string{}
	}
	return sortedCopy(task.Operation.Gaps)
}

func encodeTaskOrigin(task auditdomain.ItemTask) (json.RawMessage, error) {
	value := struct {
		SourceRef                contracts.ArtifactRef `json:"sourceRef"`
		SourceContentDigest      string                `json:"sourceContentDigest"`
		CanonicalInventoryDigest string                `json:"canonicalInventoryDigest"`
		EntryKey                 string                `json:"entryKey"`
		EntryVersion             string                `json:"entryVersion,omitempty"`
	}{
		SourceRef: task.SourceRef, SourceContentDigest: task.SourceContentDigest,
		CanonicalInventoryDigest: task.CanonicalInventoryDigest,
		EntryKey:                 task.ItemKey,
	}
	if task.Checklist != nil {
		value.EntryVersion = task.Checklist.Version
	}
	return json.Marshal(value)
}

func collectionProvenance(
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
	member preparedMember,
	run runstore.WorkflowRun,
	source auditstore.ExactArtifact,
) (json.RawMessage, error) {
	return json.Marshal(sourceProvenance{
		Schema: provenanceSchema, AuditID: snapshot.Audit.AuditID,
		ExecutionID: execution.ExecutionID, ExecutionItemID: member.member.ExecutionItemID,
		ItemID: member.item.ItemID, ItemKey: member.item.ItemKey, ItemAttempt: member.member.ItemAttempt,
		RunID: run.RunID, WorkflowName: run.WorkflowName, WorkflowVersion: run.WorkflowVersion,
		WorkflowSchemaVersion: run.WorkflowSchemaVersion, WorkflowClosureDigest: digestBytes(run.WorkflowSnapshot),
		ExecutionManifest: execution.Manifest, Task: member.member.Task,
		TaskSource: member.origin, SourceOutput: source,
	})
}

func evidenceProvenance(
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
	member preparedMember,
	run runstore.WorkflowRun,
	source auditstore.ExactArtifact,
	evidence auditdomain.Evidence,
) (json.RawMessage, error) {
	origin, err := collectionProvenance(snapshot, execution, member, run, source)
	if err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Source   json.RawMessage      `json:"source"`
		Evidence auditdomain.Evidence `json:"evidence"`
	}{Source: origin, Evidence: evidence})
}

func finalDisposition(value auditstore.CollectionDisposition) auditstore.FinalDisposition {
	switch value {
	case auditstore.CollectionMissingOutput:
		return auditstore.FinalMissingOutput
	case auditstore.CollectionInvalidResult:
		return auditstore.FinalInvalidResult
	case auditstore.CollectionExecutionFailed:
		return auditstore.FinalExecutionFailed
	case auditstore.CollectionExecutionCancelled:
		return auditstore.FinalExecutionCancelled
	case auditstore.CollectionContractInvalid:
		return auditstore.FinalInvalidResult
	default:
		return auditstore.FinalAccepted
	}
}

func technicalRationale(code string) string {
	switch code {
	case "missing-output", "result-output-contract-missing":
		return "The child Run produced no acceptable declared result output."
	case "execution-failed":
		return "The child Run failed before an acceptable semantic result was collected."
	case "execution-cancelled":
		return "The child Run was cancelled before an acceptable semantic result was collected."
	default:
		return "The child Run output failed bounded result validation."
	}
}

func stableValidationCode(err error) string {
	if code := auditdomain.ErrorCode(err); code != "" {
		return code
	}
	return auditdomain.CodeResultSetInvalid
}

func findItem(items []auditstore.Item, itemID string) (auditstore.Item, bool) {
	for _, item := range items {
		if item.ItemID == itemID {
			return item, true
		}
	}
	return auditstore.Item{}, false
}

func sameRef(left, right contracts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name &&
		left.Revision != nil && right.Revision != nil && *left.Revision == *right.Revision
}

func contains(values []string, candidate string) bool {
	for _, value := range values {
		if value == candidate {
			return true
		}
	}
	return false
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func sortedCopy(values []string) []string {
	result := append([]string{}, values...)
	sort.Strings(result)
	return result
}

func mergeSorted(left, right []string) []string {
	set := make(map[string]struct{}, len(left)+len(right))
	for _, value := range append(append([]string{}, left...), right...) {
		set[value] = struct{}{}
	}
	result := make([]string, 0, len(set))
	for value := range set {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func sortedEvidenceIDs(values map[string]validatedEvidence) []string {
	result := make([]string, 0, len(values))
	for id := range values {
		result = append(result, id)
	}
	sort.Strings(result)
	return result
}

func retainedEvidenceFits(
	audit auditstore.Audit,
	result auditstore.ExactArtifact,
	evidence map[string]validatedEvidence,
) bool {
	remaining := audit.Limits.MaxEvidenceBytes - audit.RetainedEvidenceBytes
	if remaining < 0 {
		return false
	}
	seen := make(map[string]struct{}, len(evidence)+1)
	for _, artifact := range append(
		[]auditstore.ExactArtifact{result}, externalEvidenceArtifacts(evidence)...,
	) {
		if artifact.Ref.Revision == nil || artifact.SizeBytes < 0 {
			return false
		}
		key := artifact.Ref.Namespace + "\x00" + artifact.Ref.Name + "\x00" + *artifact.Ref.Revision
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		if artifact.SizeBytes > remaining {
			return false
		}
		remaining -= artifact.SizeBytes
	}
	return true
}

func externalEvidenceArtifacts(values map[string]validatedEvidence) []auditstore.ExactArtifact {
	result := make([]auditstore.ExactArtifact, 0, len(values))
	for _, value := range values {
		if value.descriptor != nil {
			result = append(result, *value.descriptor)
		}
	}
	return result
}

func nonNilLinks(values []auditstore.ArtifactLink) []auditstore.ArtifactLink {
	if values == nil {
		return []auditstore.ArtifactLink{}
	}
	return values
}

func deterministicID(prefix string, values ...string) string {
	identity := []byte("contractor.audit.identity.v1\x00" + prefix)
	for _, value := range values {
		identity = append(identity, 0)
		identity = append(identity, value...)
	}
	return prefix + "-" + digestBytes(identity)[len("sha256:"):]
}
