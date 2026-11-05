package auditimport

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type taskOutput struct {
	logical, slot    string
	source, retained auditstore.ExactArtifact
}

// collectTasks retains ordinary declared outputs and a natural collection
// receipt. A successful supporting Run creates no semantic verdict. Logical
// result outputs alone pass through the common trusted-context normalizer.
func (i *Importer) collectTasks(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot, execution auditstore.Execution) (changed bool, err error) {
	if execution.Role != auditstore.ExecutionTask || execution.AuditID != claim.AuditID || execution.AuditID != snapshot.Audit.AuditID || execution.State != auditstore.ExecutionCollecting || execution.TerminalOutcome == nil {
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
	// A deterministic read/write rejection cannot become an endless retry of
	// the same frozen output. Preserve a technical receipt without references
	// whose validity failed. Transport/storage ambiguity keeps its original
	// error and reconciles the existing natural receipt on the next pass.
	defer func() {
		if errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
			params.Disposition = auditstore.CollectionContractInvalid
			params.SourceOutput, params.Retained = nil, nil
			changed, err = commit("evidence-budget-exhausted", false, nil)
		}
		if permanentTaskCollectionError(err) {
			params.Disposition = auditstore.CollectionContractInvalid
			params.SourceOutput, params.Retained = nil, nil
			changed, err = commit("collection-contract-invalid", false, nil)
		}
	}()
	if more, err := i.retainTaskFindings(ctx, snapshot, execution); more || err != nil {
		if errors.Is(err, ErrPermanent) {
			params.Disposition = auditstore.CollectionContractInvalid
			return commit("finding-origin-invalid", false, nil)
		}
		return more, err
	}
	gapCode := ""
	retryable := false
	switch *execution.TerminalOutcome {
	case auditstore.TerminalSubmissionFailed:
		// Retryable creation failures retain the original intent. An intent
		// is observed as failed only after definite rejection or dispatch
		// closure; neither outcome authorizes another Run attempt.
		params.Disposition = auditstore.CollectionExecutionFailed
		return commit("submission-failed", false, nil)
	case auditstore.TerminalFailed:
		params.Disposition = auditstore.CollectionExecutionFailed
		gapCode, retryable = "execution-failed", true
		if execution.RunID == nil {
			params.Disposition = auditstore.CollectionContractInvalid
			return commit("collection-contract-invalid", false, nil)
		}
		run, readErr := i.runs.GetRun(ctx, *execution.RunID)
		if readErr != nil {
			return false, readErr
		}
		if run.RunID != *execution.RunID || run.OwnerID != snapshot.Audit.OwnerID || run.ProjectID == nil || *run.ProjectID != snapshot.Audit.ProjectID || run.PublicationMode != runstore.PublicationAuditManaged || run.AuditExecutionID == nil || *run.AuditExecutionID != execution.ExecutionID || run.AuditSubmissionKey == nil || *run.AuditSubmissionKey != execution.SubmissionKey || run.State != runstore.RunFailed {
			params.Disposition = auditstore.CollectionContractInvalid
			return commit("collection-contract-invalid", false, nil)
		}
		if run.StateReason.Code == runstore.AuditInputInitializationFailedReason {
			params.Disposition = auditstore.CollectionContractInvalid
			return commit(runstore.AuditInputInitializationFailedReason, false, nil)
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
	recovered, nonrepeatable, err := i.recoverTaskScan(ctx, snapshot, execution, pinned)
	if errors.Is(err, ErrPermanent) {
		params.Disposition = auditstore.CollectionContractInvalid
		return commit("scan-history-invalid", false, nil)
	}
	if err != nil {
		return false, err
	}
	if nonrepeatable {
		retryable = false
	}
	outputs := make([]taskOutput, 0, len(binding.Outputs))
	remaining := snapshot.Audit.Limits.MaxEvidenceBytes - snapshot.Audit.RetainedEvidenceBytes
	seen := map[string]bool{}
	var sourceOutput *auditstore.ExactArtifact
	for _, logical := range slices.Sorted(maps.Keys(binding.Outputs)) {
		slot := binding.Outputs[logical]
		if recovered != nil && recovered.logical == logical {
			outputs = append(outputs, *recovered)
			remaining -= recovered.retained.SizeBytes
			if *execution.TerminalOutcome == auditstore.TerminalSucceeded {
				params.Disposition = auditstore.CollectionMissingOutput
				gapCode = "scan-report-recovered"
			}
			continue
		}
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
		if output.retained.Ref.Revision != nil {
			continue
		}
		material := auditcontract.Material{ScopeKind: "run", ScopeID: *execution.RunID, Ref: output.source.Ref, Digest: output.source.Digest, MediaType: output.source.MediaType, SizeBytes: output.source.SizeBytes}
		name, err := auditdomain.MaterialArtifactName(material)
		if err != nil {
			return false, err
		}
		retentions = append(retentions, RunRetention{Source: output.source, Target: contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(execution.AuditID), Name: name}})
	}
	retained, err := i.artifacts.RetainRunExact(ctx, *execution.RunID, snapshot.Audit.ProjectID, retentions)
	if err != nil {
		return false, err
	}
	if len(retained) != len(retentions) {
		return false, fmt.Errorf("retain Task outputs: incomplete exact result")
	}
	var selected *auditcontract.Material
	retainedIndex := 0
	for index := range outputs {
		output := &outputs[index]
		if output.retained.Ref.Revision == nil {
			output.retained = retained[retainedIndex]
			retainedIndex++
		}
		origin := map[string]any{"schema": "contractor.audit.output-provenance.v1", "audit_id": execution.AuditID, "execution_id": execution.ExecutionID, "run_id": *execution.RunID, "workflow_output": output.slot}
		if output.source.Ref.Revision == nil {
			origin["recovery"] = "retained-scan-journal"
			origin["retained"] = output.retained
		} else {
			origin["source"] = output.source
		}
		provenance, _ := strictjson.Canonical(origin)
		params.Retained = append(params.Retained, auditstore.ArtifactLink{Scope: artifacts.ScopeAudit, LogicalKey: "execution/" + execution.ExecutionID + "/output/" + output.logical, Artifact: output.retained, SourceProvenance: provenance})
		material := auditcontract.Material{ScopeKind: "audit", ScopeID: execution.AuditID, Ref: output.retained.Ref, Digest: output.retained.Digest, MediaType: output.retained.MediaType, SizeBytes: output.retained.SizeBytes}
		if selected == nil {
			copy := material
			selected = &copy
		}
		if output.logical != "result" || *execution.TerminalOutcome != auditstore.TerminalSucceeded {
			continue
		}
		data, err := i.normalizeTaskResult(ctx, snapshot.Audit, execution, pinned, tasks, *output)
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
		evidence, err := taskResultEvidenceLinks(data)
		if err != nil {
			return false, err
		}
		params.Retained = append(params.Retained, evidence...)
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
	if *execution.TerminalOutcome == auditstore.TerminalSucceeded {
		for _, output := range outputs {
			if output.logical != "work" && output.logical != "proposals" {
				continue
			}
			if i.intake == nil || output.source.MediaType != "application/json" || output.source.SizeBytes > auditcontract.MaxPreparedWorkBytes {
				params.Disposition = auditstore.CollectionInvalidResult
				params.SourceOutput = &output.source
				return commit("prepared-work-invalid", false, nil)
			}
			_, data, err := i.artifacts.ReadRunExact(ctx, *execution.RunID, output.source.Ref)
			if err != nil {
				return false, err
			}
			source := auditcontract.Material{ScopeKind: "run", ScopeID: *execution.RunID, Ref: output.source.Ref, Digest: output.source.Digest, MediaType: output.source.MediaType, SizeBytes: output.source.SizeBytes}
			accept := i.intake.AcceptPreparedWork
			if output.logical == "proposals" {
				accept = i.intake.AcceptProposals
			}
			intake, err := accept(ctx, claim, snapshot, execution, source, data)
			if err != nil {
				invalid := permanentTaskCollectionError(err) || errors.Is(err, auditstore.ErrConflict)
				if !invalid {
					return false, err
				}
				params.Disposition = auditstore.CollectionInvalidResult
				params.SourceOutput = &output.source
				return commit("prepared-work-invalid", false, nil)
			}
			if !intake.Complete {
				return intake.Changed, nil
			}
			if intake.GapCode != "" {
				gapCode = intake.GapCode
			}
		}
	}
	changed, err = commit(gapCode, retryable, selected)
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

// Cited inputs already belong to Audit scope. Link their exact revisions
// through the existing collection owner so evidence accounting charges them
// once, including when several batch members cite the same selected bytes.
func taskResultEvidenceLinks(data []byte) ([]auditstore.ArtifactLink, error) {
	var result map[string]any
	if strictjson.Decode(data, &result) != nil {
		return nil, auditstore.ErrInvalid
	}
	links := []auditstore.ArtifactLink{}
	seen := map[string]bool{}
	for _, raw := range result["items"].([]any) {
		for _, selected := range raw.(map[string]any)["evidence"].([]any) {
			encoded, err := strictjson.Canonical(selected)
			if err != nil {
				return nil, err
			}
			key := contentdigest.Bytes(encoded)
			if seen[key] {
				continue
			}
			seen[key] = true
			var material auditcontract.Material
			if strictjson.Decode(encoded, &material) != nil || material.ScopeKind != "audit" {
				return nil, auditstore.ErrInvalid
			}
			links = append(links, auditstore.ArtifactLink{Scope: artifacts.ScopeAudit, LogicalKey: "evidence/" + strings.TrimPrefix(key, "sha256:"), Artifact: contracts.ExactArtifact{Ref: material.Ref, Digest: material.Digest, MediaType: material.MediaType, SizeBytes: material.SizeBytes}, SourceProvenance: encoded})
		}
	}
	return links, nil
}

func permanentTaskCollectionError(err error) bool {
	if err == nil {
		return false
	}
	var classified apierror.Classified
	return errors.Is(err, auditstore.ErrInvalid) || errors.Is(err, blob.ErrIntegrity) || errors.Is(err, blob.ErrMissing) || artifacts.IsExactReferenceRejection(err) || artifactpostgres.IsExactRevisionMissing(err) || errors.As(err, &classified) && (classified.ErrorKind() == apierror.InvalidRequest || classified.ErrorKind() == apierror.AuditDocumentInvalid)
}

func (i *Importer) normalizeTaskResult(ctx context.Context, audit auditstore.Audit, execution auditstore.Execution, pinned auditprofile.PinnedProgram, tasks []auditstore.Task, output taskOutput) ([]byte, error) {
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
	trusted, raw, err = i.taskFindingContext(ctx, audit.OwnerID, execution, trusted, raw)
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
	inputMaterials := map[string]auditcontract.Material{}
	for _, member := range members {
		for _, raw := range member.(map[string]any)["evidence"].([]any) {
			data, err := strictjson.Canonical(raw)
			if err != nil {
				return nil, err
			}
			var m auditcontract.Material
			if strictjson.Decode(data, &m) != nil || m.ScopeKind != "run" || m.ScopeID != *execution.RunID || m.Ref.Namespace != "inputs" {
				return invalid()
			}
			if prior, exists := inputMaterials[m.Ref.Name]; exists && contracts.MaterialKey(prior) != contracts.MaterialKey(m) {
				return invalid()
			}
			inputMaterials[m.Ref.Name] = m
		}
	}
	definitions := make([]artifacts.MaterialInput, 0, len(inputMaterials))
	for name, material := range inputMaterials {
		exact := material.Exact()
		slot := pinned.Workflows[execution.WorkflowRole].Inputs[name]
		definitions = append(definitions, artifacts.MaterialInput{Name: name, Ref: material.Ref, Expected: &exact, MediaTypes: slot.MediaTypes, Set: slot.MaterialSet, Required: slot.Required})
	}
	inputDescriptions, err := i.artifacts.DescribeRunInputs(ctx, *execution.RunID, artifacts.MaterialAuthority{OwnerID: audit.OwnerID, ProjectID: audit.ProjectID, AuditID: audit.AuditID}, definitions)
	if err != nil {
		return nil, err
	}
	for index, member := range members {
		item := member.(map[string]any)
		task := tasks[index].Document
		if item["item_key"] != auditcontract.ExecutionItemKey(index) || item["task_id"] != task.TaskID || item["objective"] != task.Objective {
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
			description, found := inputDescriptions[material.Ref.Name]
			if !exists || !found || contracts.MaterialKey(original) != contracts.MaterialKey(description.Source) {
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
