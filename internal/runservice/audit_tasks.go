package runservice

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func validTaskRunIdentity(p AuditCreateParams) bool {
	if !contracts.ValidResourceID(p.ExecutionID) || p.NewRunID == nil || !contentdigest.Valid(p.RequestDigest) || !contracts.ValidResourceID(p.Claim.AuditID) || p.Claim.HolderID == "" || p.Claim.Epoch == 0 || len(p.TaskIDs) < 1 || len(p.TaskIDs) > auditstore.MaxCollectionItems {
		return false
	}
	seen := map[string]bool{}
	for _, id := range p.TaskIDs {
		if !contracts.ValidResourceID(id) || seen[id] {
			return false
		}
		seen[id] = true
	}
	return true
}

// Common Run inputs and configuration are derived exclusively from the locked
// Audit pin and immutable accepted Task documents. A coordinator passes only
// the reserved execution identity and membership, never replacement authority.
func normalizeTaskRunCreate(intent auditstore.RunCreationIntent, p AuditCreateParams) (auditRunCreate, json.RawMessage, error) {
	if !validTaskRunIdentity(p) || intent.Execution.Role != auditstore.ExecutionTask || intent.Execution.AuditID != p.Claim.AuditID || intent.Execution.ExecutionID != p.ExecutionID || intent.Execution.RequestDigest != p.RequestDigest || len(intent.Tasks) != len(p.TaskIDs) {
		return auditRunCreate{}, nil, auditstore.ErrConflict
	}
	pinned, err := auditprofile.DecodePinnedProgram(intent.ProgramSnapshot)
	if err != nil {
		return auditRunCreate{}, nil, ErrInvalid
	}
	alias := intent.Execution.WorkflowRole
	binding, ok := pinned.Program.Workflows[alias]
	if !ok {
		return auditRunCreate{}, nil, ErrInvalid
	}
	workflow, ok := pinned.Workflows[alias]
	if !ok {
		return auditRunCreate{}, nil, ErrInvalid
	}
	var baseline auditbaseline.BaselineSnapshot
	if strictjson.Decode(intent.BaselineSnapshot, &baseline) != nil || baseline.Schema != auditbaseline.Schema || baseline.RuntimeConfig.Validate() != nil {
		return auditRunCreate{}, nil, ErrInvalid
	}
	result := auditRunCreate{AuditCreateParams: AuditCreateParams{Claim: p.Claim, ExecutionID: p.ExecutionID, TaskIDs: slices.Clone(p.TaskIDs), RequestDigest: p.RequestDigest, NewRunID: p.NewRunID},
		Workflow: workflow.Clone(), RuntimeConfig: baseline.RuntimeConfig, ProjectHTTPTarget: baseline.ProjectHTTPTarget.Clone(), RequiredPlacementLabels: slices.Clone(intent.RequiredPlacementLabels),
		Inputs: map[string]auditstore.ExactArtifact{}, Parameters: maps.Clone(binding.Parameters)}
	for slot, material := range intent.Tasks[0].Document.Inputs {
		result.Inputs[slot] = contracts.ExactArtifact{Ref: material.Ref.Clone(), Digest: material.Digest, MediaType: material.MediaType, SizeBytes: material.SizeBytes}
	}
	refs, err := resolved.WorkflowSkillRefs(workflow)
	if err != nil {
		return auditRunCreate{}, nil, err
	}
	for _, ref := range refs {
		found := false
		for _, skill := range baseline.Skills {
			if skill.Name == ref.Name {
				result.Skills = append(result.Skills, skill)
				found = true
				break
			}
		}
		if !found {
			return auditRunCreate{}, nil, ErrInvalid
		}
	}
	result.Skills = agentskills.CloneRunSkills(result.Skills)
	if !matchingPinnedSkills(refs, result.Skills) {
		return auditRunCreate{}, nil, ErrInvalid
	}
	if err := agentskills.ValidateSelectedLimits(result.Skills, resolved.WorkflowSkillSets(workflow)); err != nil {
		return auditRunCreate{}, nil, err
	}
	if result.ProjectHTTPTarget != nil && result.ProjectHTTPTarget.Validate() != nil {
		return auditRunCreate{}, nil, ErrInvalid
	}
	if err := validateTaskRunIntent(intent, result); err != nil {
		return auditRunCreate{}, nil, err
	}
	snapshot, err := resolved.EncodeResolvedWorkflowSnapshot(result.Workflow)
	return result, snapshot, err
}

func validateTaskRunIntent(intent auditstore.RunCreationIntent, p auditRunCreate) error {
	if !validTaskRunIdentity(p.AuditCreateParams) || intent.Execution.Role != auditstore.ExecutionTask || intent.Execution.AuditID != p.Claim.AuditID || intent.Execution.ExecutionID != p.ExecutionID || intent.Execution.RequestDigest != p.RequestDigest || len(intent.Tasks) != len(p.TaskIDs) || len(intent.Items) != 0 {
		return auditstore.ErrConflict
	}
	pinned, err := auditprofile.DecodePinnedProgram(intent.ProgramSnapshot)
	if err != nil {
		return ErrInvalid
	}
	binding, exists := pinned.Program.Workflows[intent.Execution.WorkflowRole]
	if !exists {
		return ErrInvalid
	}
	if !slices.Equal(p.RequiredPlacementLabels, intent.RequiredPlacementLabels) || !maps.Equal(p.Parameters, binding.Parameters) {
		return auditstore.ErrConflict
	}
	selected := map[string]bool{}
	for _, id := range p.TaskIDs {
		selected[id] = true
	}
	for _, task := range intent.Tasks {
		doc := task.Document
		if !selected[doc.TaskID] || doc.AuditID != intent.Execution.AuditID || doc.Workflow != intent.Execution.WorkflowRole || !maps.Equal(doc.Parameters, binding.Parameters) || len(doc.Inputs) != len(p.Inputs) {
			return auditstore.ErrConflict
		}
		if err := auditprofile.ValidateTaskInputs(binding, p.Workflow, doc.Inputs); err != nil {
			return err
		}
		for slot, material := range doc.Inputs {
			input := p.Inputs[slot]
			if material.ScopeKind != "audit" || material.ScopeID != doc.AuditID || !material.Ref.SameExact(input.Ref) || material.Digest != input.Digest || material.MediaType != input.MediaType || material.SizeBytes != input.SizeBytes {
				return auditstore.ErrConflict
			}
		}
	}
	actual, err := contentdigest.CanonicalJSON(p.Workflow)
	if err != nil {
		return err
	}
	expected, err := contentdigest.CanonicalJSON(pinned.Workflows[intent.Execution.WorkflowRole])
	if err != nil || actual != expected {
		return auditstore.ErrConflict
	}
	return nil
}

// Context is authored after the real Run ID and input forks exist, in the same
// creation transaction. Its evidence grants access only to those local inputs.
func materializeTaskRunContext(ctx context.Context, service *artifacts.Service, intent auditstore.RunCreationIntent, p auditRunCreate, runID string, prepared *artifacts.PreparedInputs) error {
	pinned, err := auditprofile.DecodePinnedProgram(intent.ProgramSnapshot)
	if err != nil {
		return err
	}
	binding := pinned.Program.Workflows[intent.Execution.WorkflowRole]
	scope, err := artifacts.RunScope(runID)
	if err != nil {
		return err
	}
	digest, err := contentdigest.CanonicalJSON(p.Workflow)
	if err != nil {
		return err
	}
	items := make([]any, 0, len(intent.Tasks))
	for i, task := range intent.Tasks {
		evidence := make([]auditcontract.Material, 0, len(task.Document.Inputs))
		for _, slot := range sortedInputSlots(p.Inputs) {
			local := prepared.Inputs[slot]
			input := auditcontract.Material{ScopeKind: "run", ScopeID: runID, Ref: local.Ref, Digest: local.Digest, MediaType: local.MediaType, SizeBytes: local.SizeBytes}
			evidence = append(evidence, input)
		}
		item := map[string]any{"item_key": auditcontract.ExecutionItemKey(i), "task_id": task.Document.TaskID, "objective": task.Document.Objective, "evidence": evidence}
		if task.Document.Hypothesis != nil {
			if task.Hypothesis == nil || task.Hypothesis.ID != task.Document.Hypothesis.ID || task.Hypothesis.Revision != task.Document.Hypothesis.Revision {
				return ErrInvalid
			}
			item["hypothesis"] = task.Hypothesis
		}
		items = append(items, item)
	}
	for slot, mapping := range binding.Inputs {
		if mapping.Source == "audit-input" || mapping.Source == "task-input" || mapping.Source == "task-results" || mapping.Source == "prior-results" {
			continue
		}
		if mapping.Source != "task-context" {
			return fmt.Errorf("%w: prior results have no retained selection", ErrInvalid)
		}
		data, err := strictjson.Canonical(map[string]any{"schema": "contractor.audit.context.v1", "audit_id": intent.Execution.AuditID, "run_id": runID, "workflow_ref": binding.Ref, "workflow_digest": digest, "items": items})
		if err != nil {
			return err
		}
		validated, err := auditcontract.Decode(data, nil)
		if err != nil {
			return err
		}
		written, err := service.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: "inputs", Name: slot}, artifacts.Payload{MediaType: "application/json", Data: validated})
		if err != nil {
			return err
		}
		prepared.Inputs[slot] = contracts.ExactContent(written.Ref, written.MediaType, data)
	}
	if err := materializeProposalContext(ctx, service, intent, p, pinned, runID, prepared); err != nil {
		return err
	}
	return validateWorkflowInputs(p.Workflow, p.Parameters, prepared.Refs())
}
