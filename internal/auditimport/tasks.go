package auditimport

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

type taskOutput struct {
	logical, slot    string
	source, retained auditstore.ExactArtifact
}

// collectTasks retains ordinary declared outputs and a natural collection
// receipt. A successful supporting Run creates no semantic verdict. Logical
// result outputs alone pass through the common trusted-context normalizer.
func (i *Importer) collectTasks(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot, execution auditstore.Execution) (bool, error) {
	if execution.AuditID != claim.AuditID || execution.AuditID != snapshot.Audit.AuditID || execution.State != auditstore.ExecutionCollecting || execution.TerminalOutcome == nil {
		return false, fmt.Errorf("%w: invalid Task collection observation", ErrPermanent)
	}
	digest, err := contentdigest.CanonicalJSON(map[string]any{"schema": "contractor.audit.task-collection.v1", "audit_id": execution.AuditID, "execution_id": execution.ExecutionID, "run_id": execution.RunID, "outcome": execution.TerminalOutcome, "generation": execution.TerminalRunGeneration, "sequence": execution.TerminalRunSequence})
	if err != nil {
		return false, err
	}
	if receipt, err := i.reads.CollectionReceipt(ctx, execution.AuditID, execution.ExecutionID); err == nil {
		if receipt.RequestDigest != digest {
			return false, auditstore.ErrConflict
		}
		return false, nil
	} else if !errors.Is(err, auditstore.ErrNotFound) {
		return false, err
	}
	tasks, err := i.reads.ListExecutionTasks(ctx, execution.AuditID, execution.ExecutionID)
	if err != nil {
		return false, err
	}
	params := auditstore.CollectParams{Claim: claim, ExecutionID: execution.ExecutionID, ReceiptID: auditdomain.DeterministicID("receipt", execution.ExecutionID), RequestDigest: digest, Disposition: auditstore.CollectionAccepted, Retained: []auditstore.ArtifactLink{}}
	commit := func(code string, retryable bool, result *auditcontract.Material) (bool, error) {
		params.Tasks = make([]auditstore.TaskCollection, 0, len(tasks))
		for _, task := range tasks {
			var gap *string
			if code != "" {
				value := code
				gap = &value
			}
			params.Tasks = append(params.Tasks, auditstore.TaskCollection{ExecutionItemID: auditdomain.DeterministicID("attempt", execution.ExecutionID, task.Document.TaskID), TaskID: task.Document.TaskID, Result: result, GapCode: gap, Retryable: retryable})
		}
		if code != "" {
			params.ErrorCode = &code
		}
		_, changed, err := i.store.Collect(ctx, params)
		return changed, err
	}
	gapCode := ""
	retryable := false
	switch *execution.TerminalOutcome {
	case auditstore.TerminalFailed, auditstore.TerminalSubmissionFailed:
		params.Disposition = auditstore.CollectionExecutionFailed
		gapCode, retryable = "execution-failed", true
		if execution.RunID == nil {
			return commit(gapCode, retryable, nil)
		}
	case auditstore.TerminalCancelled:
		params.Disposition = auditstore.CollectionExecutionCancelled
		gapCode = "execution-cancelled"
	case auditstore.TerminalSucceeded:
	default:
		return false, fmt.Errorf("%w: unknown Task terminal outcome", ErrPermanent)
	}
	pinned, err := auditprofile.DecodePinnedProgram(snapshot.Audit.ProfileSnapshot)
	if err != nil || execution.RunID == nil {
		params.Disposition = auditstore.CollectionContractInvalid
		return commit("collection-contract-invalid", false, nil)
	}
	binding, exists := pinned.Program.Workflows[execution.WorkflowRole]
	if !exists {
		params.Disposition = auditstore.CollectionContractInvalid
		return commit("collection-contract-invalid", false, nil)
	}
	outputs := make([]taskOutput, 0, len(binding.Outputs))
	remaining := snapshot.Audit.Limits.MaxEvidenceBytes - snapshot.Audit.RetainedEvidenceBytes
	seen := map[string]bool{}
	var sourceOutput *auditstore.ExactArtifact
	for _, logical := range slices.Sorted(maps.Keys(binding.Outputs)) {
		slot := binding.Outputs[logical]
		source, frozen, err := i.artifacts.DescribeRunBinding(ctx, *execution.RunID, contracts.ArtifactRef{Namespace: "outputs", Name: slot})
		if errors.Is(err, artifacts.ErrArtifactNotFound) {
			if gapCode == "" {
				params.Disposition = auditstore.CollectionMissingOutput
				gapCode = "missing-output"
			}
			continue
		}
		if err != nil {
			return false, err
		}
		if !frozen || !contracts.AcceptsMediaType(pinned.Workflows[execution.WorkflowRole].Outputs[slot].MediaTypes, source.MediaType) {
			if *execution.TerminalOutcome == auditstore.TerminalSucceeded {
				params.Disposition = auditstore.CollectionContractInvalid
				gapCode = "collection-contract-invalid"
			}
			continue
		}
		key, err := contentdigest.CanonicalJSON(source)
		if err != nil {
			return false, err
		}
		if !seen[key] {
			remaining -= source.SizeBytes
			seen[key] = true
		}
		if sourceOutput == nil || logical == "result" {
			copy := source
			sourceOutput = &copy
		}
		if remaining < 0 {
			if *execution.TerminalOutcome == auditstore.TerminalSucceeded {
				params.Disposition = auditstore.CollectionInvalidResult
				params.SourceOutput = sourceOutput
			}
			return commit("evidence-budget-exhausted", false, nil)
		}
		outputs = append(outputs, taskOutput{logical: logical, slot: slot, source: source})
	}
	if params.Disposition == auditstore.CollectionAccepted {
		params.SourceOutput = sourceOutput
	}
	retentions := make([]RunRetention, 0, len(outputs))
	for _, output := range outputs {
		material := auditcontract.Material{ScopeKind: "run", ScopeID: *execution.RunID, Ref: output.source.Ref, Digest: output.source.Digest, MediaType: output.source.MediaType, SizeBytes: output.source.SizeBytes}
		name, err := auditdomain.MaterialArtifactName(material)
		if err != nil {
			return false, err
		}
		retentions = append(retentions, RunRetention{AuditOwned: true, Source: output.source, Target: contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(execution.AuditID), Name: name}})
	}
	retained, err := i.artifacts.RetainRunExact(ctx, *execution.RunID, snapshot.Audit.ProjectID, retentions)
	if err != nil {
		return false, err
	}
	if len(retained) != len(outputs) {
		return false, fmt.Errorf("retain Task outputs: incomplete exact result")
	}
	var selected *auditcontract.Material
	for index := range outputs {
		output := &outputs[index]
		output.retained = retained[index]
		provenance, _ := strictjson.Canonical(map[string]any{"schema": "contractor.audit.output-provenance.v1", "audit_id": execution.AuditID, "execution_id": execution.ExecutionID, "run_id": *execution.RunID, "workflow_output": output.slot, "source": output.source})
		params.Retained = append(params.Retained, auditstore.ArtifactLink{Scope: artifacts.ScopeAudit, LogicalKey: "execution/" + execution.ExecutionID + "/output/" + output.logical, Artifact: output.retained, SourceProvenance: provenance})
		material := auditcontract.Material{ScopeKind: "audit", ScopeID: execution.AuditID, Ref: output.retained.Ref, Digest: output.retained.Digest, MediaType: output.retained.MediaType, SizeBytes: output.retained.SizeBytes}
		if selected == nil {
			copy := material
			selected = &copy
		}
		if output.logical != "result" || *execution.TerminalOutcome != auditstore.TerminalSucceeded {
			continue
		}
		data, err := i.normalizeTaskResult(ctx, execution, pinned, tasks, *output)
		if errors.Is(err, ErrPermanent) {
			params.Disposition = auditstore.CollectionInvalidResult
			params.SourceOutput = sourceOutput
			return commit("invalid-result", false, nil)
		}
		if err != nil {
			return false, err
		}
		name := "result-" + strings.TrimPrefix(contentdigest.Bytes(data), "sha256:")
		result, err := i.artifacts.PutImmutableAudit(ctx, snapshot.Audit.ProjectID, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(execution.AuditID), Name: name}, artifacts.Payload{MediaType: "application/json", Data: data})
		if err != nil {
			return false, err
		}
		params.Retained = append(params.Retained, auditstore.ArtifactLink{Scope: artifacts.ScopeAudit, LogicalKey: "execution/" + execution.ExecutionID + "/result", Artifact: result, SourceProvenance: provenance})
		material = auditcontract.Material{ScopeKind: "audit", ScopeID: execution.AuditID, Ref: result.Ref, Digest: result.Digest, MediaType: result.MediaType, SizeBytes: result.SizeBytes}
		selected = &material
	}
	if selected == nil {
		if gapCode == "" {
			params.Disposition = auditstore.CollectionMissingOutput
			gapCode = "missing-output"
		}
		return commit(gapCode, retryable, nil)
	}
	changed, err := commit(gapCode, retryable, selected)
	if errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
		params.Retained = nil
		if *execution.TerminalOutcome == auditstore.TerminalSucceeded {
			params.Disposition = auditstore.CollectionInvalidResult
			params.SourceOutput = sourceOutput
		}
		return commit("evidence-budget-exhausted", false, nil)
	}
	return changed, err
}

func (i *Importer) normalizeTaskResult(ctx context.Context, execution auditstore.Execution, pinned auditprofile.PinnedProgram, tasks []auditstore.Task, output taskOutput) ([]byte, error) {
	invalid := func() ([]byte, error) { return nil, fmt.Errorf("%w: invalid common Task result", ErrPermanent) }
	if output.source.MediaType != "application/json" || output.source.SizeBytes > auditcontract.MaximumDocumentBytes {
		return invalid()
	}
	binding := pinned.Program.Workflows[execution.WorkflowRole]
	contextSlot := ""
	for slot, mapping := range binding.Inputs {
		if mapping.Source == "task-context" {
			contextSlot = slot
			break
		}
	}
	if contextSlot == "" {
		return invalid()
	}
	exact, frozen, err := i.artifacts.DescribeRunBinding(ctx, *execution.RunID, contracts.ArtifactRef{Namespace: "inputs", Name: contextSlot})
	if err != nil {
		return nil, err
	}
	if !frozen || exact.SizeBytes > auditcontract.MaximumDocumentBytes || exact.MediaType != "application/json" {
		return invalid()
	}
	_, trusted, err := i.artifacts.ReadRunExact(ctx, *execution.RunID, exact.Ref)
	if err != nil {
		return nil, err
	}
	if _, err := auditcontract.Decode(trusted, nil); err != nil {
		return invalid()
	}
	_, raw, err := i.artifacts.ReadRunExact(ctx, *execution.RunID, output.source.Ref)
	if err != nil {
		return nil, err
	}
	result, err := auditcontract.Decode(raw, trusted)
	if err != nil {
		return invalid()
	}
	var contextValue, resultValue map[string]any
	if strictjson.Decode(trusted, &contextValue) != nil || strictjson.Decode(result, &resultValue) != nil {
		return invalid()
	}
	workflowDigest, err := contentdigest.CanonicalJSON(pinned.Workflows[execution.WorkflowRole])
	if err != nil {
		return nil, err
	}
	if contextValue["audit_id"] != execution.AuditID || contextValue["run_id"] != *execution.RunID || contextValue["workflow_ref"] != binding.Ref || contextValue["workflow_digest"] != workflowDigest {
		return invalid()
	}
	members := contextValue["items"].([]any)
	if len(members) != len(tasks) {
		return invalid()
	}
	translations := map[string]auditcontract.Material{}
	for index, member := range members {
		item := member.(map[string]any)
		task := tasks[index].Document
		if item["task_id"] != task.TaskID || item["objective"] != task.Objective {
			return invalid()
		}
		for _, raw := range item["evidence"].([]any) {
			data, err := strictjson.Canonical(raw)
			if err != nil {
				return nil, err
			}
			var material auditcontract.Material
			if strictjson.Decode(data, &material) != nil || material.ScopeKind != "run" || material.ScopeID != *execution.RunID || material.Ref.Namespace != "inputs" {
				return invalid()
			}
			original, exists := task.Inputs[material.Ref.Name]
			if !exists || original.Digest != material.Digest || original.MediaType != material.MediaType || original.SizeBytes != material.SizeBytes {
				return invalid()
			}
			translations[string(data)] = original
		}
	}
	translate := func(items []any) bool {
		for _, raw := range items {
			item := raw.(map[string]any)
			evidence := []any{}
			seen := map[string]bool{}
			for _, raw := range item["evidence"].([]any) {
				data, err := strictjson.Canonical(raw)
				if err != nil {
					return false
				}
				material, exists := translations[string(data)]
				if !exists {
					return false
				}
				converted, err := strictjson.Canonical(material)
				if err != nil {
					return false
				}
				if !seen[string(converted)] {
					seen[string(converted)] = true
					evidence = append(evidence, material)
				}
			}
			item["evidence"] = evidence
		}
		return true
	}
	if !translate(members) || !translate(resultValue["items"].([]any)) {
		return invalid()
	}
	contextData, err := strictjson.Canonical(contextValue)
	if err != nil {
		return nil, err
	}
	resultData, err := strictjson.Canonical(resultValue)
	if err != nil {
		return nil, err
	}
	normalized, err := auditcontract.Decode(resultData, contextData)
	if err != nil {
		return invalid()
	}
	return normalized, nil
}
