package auditservice

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

type typedCandidateResult struct {
	item          auditdomain.PreparedRoundItem
	retained      []auditstore.ArtifactLink
	pageContent   []byte
	admitted      bool
	unschedulable string
}

// prepareTypedCandidate joins the session's candidate savepoint. The complete
// source, dependency, task and page tuple exists before budget admission; a
// refused candidate rolls all of that speculative material back together.
func prepareTypedCandidate(ctx context.Context, service *artifacts.Service, reads auditstore.AdmissionReader, audit auditstore.Audit, profile auditprofile.ResolvedAuditProfile, baseline BaselineSnapshot, selection DraftSelection, progress auditstore.AdmissionProgress, routes *RoutedCheckContext, budget *auditdomain.FindingRoundBudget, check auditstore.AdmissionFindingCheck, ordinal int) (typedCandidateResult, error) {
	var result typedCandidateResult
	route, ok := routes.selectBinding(check.Plan)
	if !ok {
		return result, fmt.Errorf("%w: typed check method is unmapped", ErrInvalid)
	}
	binding := profile.Workflows[route.route.WorkflowRole]
	if check.Source.EvidenceLimitExceeded && hasRetainedDependencyMapping(binding) {
		return result, auditstore.ErrDependencyClosureLimit
	}
	itemKey := auditdomain.FindingCheckItemKey(check.Source.ReceiptID, ordinal)
	subjectKey := check.Source.ReceiptID
	if check.Context.Subject != nil {
		subjectKey = check.Context.Subject.Key
	}
	receipt := findingintake.Receipt{ReceiptID: check.Source.ReceiptID, Proposal: check.Source.Original, Evidence: slices.Clone(check.Source.OriginalEvidence), Origin: findingintake.Origin{RunID: check.Source.OriginRunID}, AuditHolds: []findingintake.AuditHold{{AuditID: audit.AuditID, ProjectID: audit.ProjectID, Proposal: check.Source.Retained, Evidence: slices.Clone(check.Source.RetainedEvidence)}}}
	if check.Source.OriginAuditID != "" {
		receipt.Origin.Audit = &findingintake.AuditOrigin{AuditID: check.Source.OriginAuditID, ExecutionID: check.Source.OriginExecutionID}
	}
	// Only the complete source factory below publishes a task. This
	// internal descriptor supplies the existing trusted dependency constructor.
	task := auditdomain.ItemTask{ItemKey: itemKey, SubjectKey: subjectKey, WorkflowRole: route.route.WorkflowRole, Finding: &auditdomain.FindingTask{ReceiptID: check.Source.ReceiptID, ProposedCheckOrdinal: ordinal}}
	writer := auditArtifactWriter{service: service, projectID: audit.ProjectID}
	manifests, links, _, err := buildAcceptedDependenciesWithReader(ctx, writer, reads, audit, profile, baseline, auditdomain.Inventory{Tasks: []auditdomain.GeneratedTask{{Document: task}}}, map[string]findingintake.Receipt{receipt.ReceiptID: receipt})
	if err != nil {
		return result, err
	}
	var dependencies *auditdomain.RetainedDependencyManifest
	if manifest, ok := manifests[itemKey]; ok {
		dependencies = &manifest
	}
	prepared, err := typedPreparationInputs(ctx, reads, audit.AuditID, binding)
	if err != nil {
		return result, err
	}
	if err := validateTypedInputMedia(binding, selection, prepared, dependencies); err != nil {
		return result, err
	}
	source := auditdomain.AcceptedPlanSource{Kind: "finding-receipt", ReceiptID: receipt.ReceiptID, ProposedCheckOrdinal: &ordinal, RunID: receipt.Origin.RunID, Proposal: check.Source.Original, RetainedProposal: check.Source.Retained}
	accepted, err := routes.build(source, check.Plan, dependencies)
	if err != nil {
		return result, err
	}
	document := auditdomain.FindingCheckSourceDocument{Schema: auditdomain.FindingCheckSourceSchema, Source: source, Check: check.Check, Subject: check.Context.Subject, Limitations: slices.Clone(check.Context.Limitations)}
	sourceBytes, err := auditdomain.EncodeFindingCheckSourceDocument(document)
	if err != nil {
		return result, err
	}
	namespace := auditdomain.ArtifactNamespace(audit.AuditID)
	sourceArtifact, err := writer.write(ctx, contracts.ArtifactRef{Namespace: namespace, Name: auditdomain.DeterministicID("finding-check-source", audit.AuditID, auditdomain.DigestBytes(sourceBytes))}, artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: sourceBytes})
	if err != nil {
		return result, nextRoundWriteError(err)
	}
	options := auditdomain.FindingCheckInventoryOptions{Common: nextRoundInventoryOptions(progress.RoundOrdinal, profile, baseline, route.authorization.ApprovalRequirement, sourceArtifact.Ref), Check: auditdomain.FindingCheckOptions{WorkflowRole: route.route.WorkflowRole, ApprovalRequirement: route.authorization.ApprovalRequirement, ExecutionInputs: acceptedTaskInputs(binding, selection, dependencies, prepared), AcceptedPlan: &accepted, Dependencies: dependencies}}
	inventory, err := auditdomain.BuildFindingCheckInventory(sourceBytes, options)
	if err != nil {
		return result, err
	}
	if err := validateInventoryTaskExecution(profile, inventory); err != nil {
		return result, err
	}
	tasks, manifest, err := writeTaskPackages(ctx, writer, namespace, profile, selection, inventory, prepared)
	if err != nil {
		return result, err
	}
	itemID := auditdomain.DeterministicID("item", audit.AuditID, fmt.Sprint(progress.RoundOrdinal), itemKey)
	work := inventory.Worklist.Items[0]
	kind, digest, state, err := materializedItemApproval(audit.AuditID, profile, itemID, work, tasks[0])
	if err != nil {
		return result, err
	}
	result.item = auditdomain.PreparedRoundItem{ItemID: itemID, ItemKey: itemKey, Ordinal: progress.SelectedCount, Kind: work.Kind, SubjectKey: work.SubjectKey, WorkflowRole: work.WorkflowRole, Task: tasks[0], InitialState: string(state), ApprovalKind: string(kind), ApprovalDigest: digest,
		Origin:          auditdomain.PreparedItemOrigin{Schema: auditdomain.ItemOriginSchema, SourceRef: sourceArtifact.Ref, SourceContentDigest: inventory.SourceContentDigest, SourceMediaType: auditdomain.JSONMediaType, CanonicalInventoryDigest: inventory.CanonicalInventoryDigest, EntryKey: itemKey},
		Coverage:        auditdomain.RoundResultCoverage{Status: auditdomain.CoverageNotTested, Requested: slices.Clone(inventory.Coverage.Rows[0].Requested), Completed: []string{}, Gaps: slices.Clone(inventory.Coverage.Rows[0].Gaps)},
		ProposalSources: []auditdomain.PreparedProposalSource{{ReceiptID: receipt.ReceiptID, ProposedCheckOrdinal: ordinal, Proposal: check.Source.Retained}}, Execution: auditdomain.PreparedExecutionItem{ExecutionItem: manifest.Items[0], AcceptedPlan: inventory.Tasks[0].Document.AcceptedPlan, Dependencies: inventory.Tasks[0].Document.Dependencies}}
	page := typedCandidatePage(progress, result.item)
	result.pageContent, err = auditdomain.EncodePreparedRoundPage(page)
	if err != nil {
		return result, err
	}
	stageBytes, err := hookstore.PreparedPageStageBytes(page)
	if err != nil {
		return result, err
	}
	projectionBytes, err := auditdomain.EncodeFindingChecksProjection(check.Source.Projection)
	if err != nil {
		return result, err
	}
	sourceMetadata, err := json.Marshal(struct {
		Original, Retained                 auditstore.ExactArtifact
		OriginalEvidence, RetainedEvidence []auditstore.ExactArtifact
	}{check.Source.Original, check.Source.Retained, check.Source.OriginalEvidence, check.Source.RetainedEvidence})
	if err != nil {
		return result, err
	}
	// The later staging transaction reads this same immutable page, validates
	// its exact source index and writes the actual row envelope. Reserve the
	// bounded progress document too, before selecting the candidate.
	if len(result.pageContent)+stageBytes+len(projectionBytes)+len(sourceMetadata)+256 > auditstore.MaxAdmissionMetadataBytes-auditstore.MaxAdmissionProgressBytes {
		return typedCandidateResult{unschedulable: "prepared_page.work"}, nil
	}
	options.Check.PreparedMetadataBytes = int64(len(result.pageContent))
	admission, err := budget.AdmitCheck(sourceBytes, options)
	if err != nil {
		return result, err
	}
	if !admission.Admitted {
		return typedCandidateResult{unschedulable: admission.Unschedulable}, nil
	}
	result.retained = links
	for _, generated := range []struct {
		kind     string
		artifact auditstore.ExactArtifact
	}{{"source", sourceArtifact}, {"task", tasks[0]}} {
		provenance, err := json.Marshal(map[string]string{"schema": auditstore.AdmissionProvenanceSchema, "generation": progress.Generation, "kind": generated.kind, "item_id": itemID})
		if err != nil {
			return result, err
		}
		result.retained = append(result.retained, auditstore.ArtifactLink{LogicalKey: "admission/" + progress.Generation + "/" + generated.kind + "/" + itemID, Artifact: generated.artifact, SourceProvenance: provenance})
	}
	result.admitted = true
	return result, nil
}

func typedCandidatePage(progress auditstore.AdmissionProgress, item auditdomain.PreparedRoundItem) auditdomain.PreparedRoundPage {
	return auditdomain.PreparedRoundPage{Schema: auditdomain.PreparedRoundPageSchema, AuditID: progress.AuditID, RoundID: progress.RoundID, Generation: progress.Generation, PageOrdinal: progress.PublishedPages, FirstItemOrdinal: item.Ordinal, PreviousPageDigest: progress.PagesDigest, Items: []auditdomain.PreparedRoundItem{item}}
}

func typedPreparationInputs(ctx context.Context, reads dependencySourceReader, auditID string, binding auditprofile.ResolvedAuditWorkflowBinding) (map[string]auditstore.ExactArtifact, error) {
	result := map[string]auditstore.ExactArtifact{}
	for slot, mapping := range binding.Inputs {
		if mapping.Source != auditprofile.AuditInputFromPreparation {
			continue
		}
		key := preparationInputKey(mapping)
		if _, ok := result[key]; ok {
			continue
		}
		output, err := reads.GetPreparationOutput(ctx, auditID, mapping.Role, mapping.Name)
		if err != nil {
			return nil, err
		}
		if !contracts.AcceptsMediaType(binding.Workflow.Inputs[slot].MediaTypes, output.Output.Retained.MediaType) {
			return nil, fmt.Errorf("%w: typed preparation input media differs", ErrInvalid)
		}
		result[key] = output.Output.Retained.Clone()
	}
	return result, nil
}

func validateTypedInputMedia(binding auditprofile.ResolvedAuditWorkflowBinding, selection DraftSelection, prepared map[string]auditstore.ExactArtifact, deps *auditdomain.RetainedDependencyManifest) error {
	for slot, mapping := range binding.Inputs {
		var descriptor auditstore.ExactArtifact
		var present bool
		switch mapping.Source {
		case auditprofile.AuditInputFromAudit:
			descriptor, present = selection.Inputs[mapping.Name]
		case auditprofile.AuditInputFromPreparation:
			descriptor, present = prepared[preparationInputKey(mapping)]
		case auditprofile.AuditInputFromRetainedDependency:
			if deps != nil {
				for _, dependency := range deps.Dependencies {
					if dependency.Name == mapping.Name {
						descriptor, present = dependency.Retained, true
						break
					}
				}
			}
		default:
			continue
		}
		contract := binding.Workflow.Inputs[slot]
		if !present {
			if contract.Required {
				return fmt.Errorf("%w: typed required input is absent", ErrInvalid)
			}
			continue
		}
		if descriptor.Validate() != nil || !descriptor.HasMetadata() || !contracts.AcceptsMediaType(contract.MediaTypes, descriptor.MediaType) {
			return fmt.Errorf("%w: typed input media or exact metadata differs", ErrInvalid)
		}
	}
	return nil
}
