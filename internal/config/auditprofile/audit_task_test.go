package auditprofile_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestAuditTaskAllowsSurroundingStagesAndOtherRoles(t *testing.T) {
	t.Parallel()
	snapshot := loadCatalog(t, copyCatalog(t, "audit-scan-catalog"))
	profile, err := snapshot.AuditProfile("openapi-sqlmap-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	ordinary, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	binding := profile.Workflows["scan"]
	workflow := &binding.Workflow
	notes := resolved.ArtifactSlot{Required: true, MediaTypes: []string{"text/plain"}}
	profile.Inputs["operator_notes"] = auditprofile.AuditProfileInput{Required: true, MediaTypes: notes.MediaTypes}
	workflow.Inputs["notes"] = notes
	binding.Inputs["notes"] = auditprofile.AuditWorkflowInputMapping{Source: auditprofile.AuditInputFromAudit, Name: "operator_notes"}
	for _, name := range []string{"before", "after"} {
		stage := ordinary.Clone().Stages[ordinary.EntryStage]
		stage.Context = resolved.StageContext{Artifacts: map[string]resolved.ContextArtifact{
			"notes": {Namespace: "inputs", Name: "notes", Required: true},
		}}
		stage.WorkflowOutputs = map[string]string{}
		stage.Result.Artifacts["copied"].From.Name = name
		workflow.Stages[name] = stage
	}
	before := workflow.Stages["before"]
	before.On.Succeeded = resolved.TransitionAction{Kind: resolved.TransitionNext, NextStage: "scan"}
	workflow.Stages["before"] = before
	scan := workflow.Stages["scan"]
	scan.On.Succeeded = resolved.TransitionAction{Kind: resolved.TransitionNext, NextStage: "after"}
	workflow.Stages["scan"] = scan
	workflow.EntryStage = "before"
	profile.Workflows["scan"] = binding
	// Other check roles declare their own consumer contracts; they do not
	// become the inventory's default route merely by accepting the same type.
	profile.Workflows["second-check"] = binding
	// A separate existing Workflow role is unrelated to task execution.
	profile.Workflows["assessment"] = auditprofile.ResolvedAuditWorkflowBinding{
		Kind: auditprofile.AuditWorkflowAssessment, Workflow: ordinary,
		Inputs: map[string]auditprofile.AuditWorkflowInputMapping{"source": {Source: auditprofile.AuditInputFromAudit, Name: "operator_notes"}}, Parameters: map[string]auditprofile.AuditWorkflowParameterMapping{},
		Outputs: map[string]string{"report": "result"},
	}
	if err := resolved.ValidateWorkflowGraph(*workflow); err != nil {
		t.Fatal(err)
	}
	if err := auditprofile.ValidateAuditTaskProfile(profile); err != nil {
		t.Fatal(err)
	}
	redigest(t, &profile)
	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := auditprofile.DecodeResolvedAuditProfileSnapshot(encoded)
	if err != nil || restored.Workflows["scan"].Workflow.AuditTask.Contract != contracts.AuditTaskOpenAPIScanV1 {
		t.Fatalf("composed Workflow cannot be restored: %v", err)
	}
	// Neither another result producer nor a success branch around the scan
	// may turn the composition into a fabricated successful task result.
	other := workflow.Clone().Stages["scan"]
	other.AuditScan, other.ScanPlan = nil, nil
	other.WorkflowOutputs = map[string]string{"result": "report"}
	changed := profile.Clone()
	changed.Workflows["scan"].Workflow.Stages["after"] = other
	if auditprofile.ValidateAuditTaskProfile(changed) == nil {
		t.Fatal("accepted a second canonical result producer")
	}
	changed = profile.Clone()
	bypass := changed.Workflows["scan"].Workflow.Stages["before"]
	bypass.On.Succeeded = resolved.TransitionAction{Kind: resolved.TransitionSucceed}
	// Keep the executor reachable, so rejection must cover the successful
	// bypass path rather than merely detect an unreachable Stage.
	bypass.On.Failed = resolved.TransitionAction{Kind: resolved.TransitionNext, NextStage: "scan"}
	changed.Workflows["scan"].Workflow.Stages["before"] = bypass
	if resolved.ValidateWorkflowGraph(changed.Workflows["scan"].Workflow) == nil {
		t.Fatal("accepted a successful path without the task result")
	}
}
