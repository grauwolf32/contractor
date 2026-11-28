package runservice

import (
	"context"
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// A concrete public caller of the existing completion resolver: one HTTP
// Worker, fixed generated inputs and no automatic repetition of target work.
func findingCompletionTarget(w resolved.ResolvedWorkflow, p PublicCreateParams) (*auditprofile.WorkerCompletionTarget, error) {
	selected := false
	for _, stage := range w.Stages {
		for _, agent := range stage.Agents {
			selected = selected || agent.Template.SelectsAuditCompletion()
		}
	}
	if p.FindingCheck == nil && !selected {
		return nil, nil
	}
	if p.FindingCheck == nil || len(w.Stages) != 1 || len(w.Inputs) != 2 || len(w.Parameters) != 1 || !w.Parameters["target"].Required || len(p.Parameters) != 0 || len(p.Inputs) != 0 {
		return nil, fmt.Errorf("%w: Finding validation requires its selection and compatible single HTTP Workflow", ErrInvalid)
	}
	for _, name := range []string{"context", "finding"} {
		slot, found := w.Inputs[name]
		if !found || !slot.Required || slot.MaterialSet || slot.SourceScope == "user" || !slot.AcceptsInputMediaType("application/json") {
			return nil, ErrInvalid
		}
	}
	stage := w.Stages[w.EntryStage]
	if stage.On.Failed.Kind != resolved.TransitionFail || stage.On.Interrupted.Kind != resolved.TransitionFail {
		return nil, ErrInvalid
	}
	http := false
	for _, agent := range stage.Agents {
		for _, selection := range agent.Template.Toolsets {
			if slices.Contains(selection.Tools, "finding") {
				return nil, ErrInvalid
			}
			http = http || selection.Ref.ToolsetID == "http-tools" && selection.Ref.Version == "1" && slices.Contains(selection.Tools, "http_request")
		}
	}
	binding := auditcontract.ProgramWorkflow{Inputs: map[string]auditcontract.InputMapping{"context": {Source: "task-context"}}, Outputs: map[string]string{"result": "result"}}
	target, err := auditprofile.ResolveCompletionTarget(w, binding)
	if err != nil || target == nil || !http {
		return nil, fmt.Errorf("%w: incompatible Finding validation completion", ErrInvalid)
	}
	return target, nil
}

func materializeFindingCheck(ctx context.Context, writer PublicRunWriter, service *artifacts.Service, run runstore.WorkflowRun, workflow resolved.ResolvedWorkflow, check contracts.FindingCheck, target auditprofile.WorkerCompletionTarget) error {
	owner, ok := writer.(FindingCheckWriter)
	if !ok {
		return ErrNotConfigured
	}
	selected, err := owner.PrepareFindingCheck(ctx, run.RunID, check)
	if err != nil {
		return err
	}
	project, _ := artifacts.ProjectScope(*run.ProjectID)
	local, _ := artifacts.RunScope(run.RunID)
	sources := append([]contracts.ExactArtifact{*selected.ProjectProposal}, selected.ProjectEvidence...)
	evidence := make([]auditcontract.Material, 0, len(selected.ProjectEvidence))
	for i, source := range sources {
		name := "finding"
		if i > 0 {
			name = fmt.Sprintf("finding-evidence-%03d", i)
		}
		fork, err := service.Fork(ctx, artifacts.Fork{Source: project, SourceRef: source.Ref, Target: local, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: name}, Kind: artifacts.LineageInputFork, Frozen: true, Replay: true})
		if err != nil {
			return err
		}
		if i > 0 {
			evidence = append(evidence, auditcontract.Material{ScopeKind: "run", ScopeID: run.RunID, Ref: fork.TargetRef, Digest: source.Digest, MediaType: source.MediaType, SizeBytes: source.SizeBytes})
		}
	}
	digest, err := contentdigest.CanonicalJSON(workflow)
	if err != nil {
		return err
	}
	data, err := strictjson.Canonical(map[string]any{"schema": "contractor.audit.context.v1", "project_id": *run.ProjectID, "finding": check, "run_id": run.RunID, "workflow_ref": workflow.Ref.Name + "@" + workflow.Ref.Version, "workflow_digest": digest, "items": []any{map[string]any{"item_key": auditcontract.ExecutionItemKey(0), "finding_id": check.FindingID, "objective": check.Assertion, "evidence": evidence}}})
	if err != nil {
		return err
	}
	contextInput, err := service.CreateFrozen(ctx, local, contracts.ArtifactRef{Namespace: "inputs", Name: target.ContextSlot}, artifacts.Payload{MediaType: "application/json", Data: data})
	if err != nil {
		return err
	}
	return owner.SetAuditCompletion(ctx, run.RunID, runstore.AuditCompletionSnapshot{Finding: &check, Stage: target.Stage, Agent: target.Agent, Contract: contracts.WorkerCompletionContract{Kind: contracts.AuditCheckResultsV1, Context: contextInput.Ref, ResultArtifact: target.Result}})
}
