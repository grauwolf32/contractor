package scheduler

import (
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestAuditCompletionTargetAndPinnedInputs(t *testing.T) {
	h := newSchedulerHarness(t)
	configureEscalationWorkflow(t, h, "failed", 2)
	workflow, err := decodeExecutableWorkflow(h.store.run)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.stage
	var agentName string
	for name, agent := range stage.Agents {
		agentName = name
		agent.Template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "audit-results", Version: "2"}, Tools: []string{"read_audit_task", "submit_check_result"}}}
		stage.Agents[name] = agent
	}
	workflow.stage = stage
	workflow.workflow.Stages[workflow.stageName] = stage
	var output contracts.ArtifactRef
	for _, result := range stage.Result.Artifacts {
		if result.From != nil {
			output = contracts.ArtifactRef{Namespace: result.From.Namespace, Name: result.From.Name}
		}
	}
	run := h.store.run
	executionID, key := "execution", "submission"
	run.PublicationMode = runstore.PublicationAuditManaged
	run.AuditExecutionID, run.AuditSubmissionKey = &executionID, &key
	run.AuditCompletion = &runstore.AuditCompletionSnapshot{Stage: workflow.stageName, Agent: agentName, Contract: contracts.WorkerCompletionContract{
		Kind: contracts.AuditCheckResultsV1, Task: exactRef("inputs", "task", "task-r1"), ExecutionManifest: exactRef("inputs", "manifest", "manifest-r1"), ResultArtifact: output}}
	pinned := runstore.StageContextSnapshot{Artifacts: map[string]runstore.PinnedContextArtifact{
		"task": {Required: true, Artifact: &run.AuditCompletion.Contract.Task}, "manifest": {Required: true, Artifact: &run.AuditCompletion.Contract.ExecutionManifest}}}
	bindings, err := auditBindingRequirements(run, workflow, pinned)
	if err != nil || !reflect.DeepEqual(bindings[0].CompletionContract, &run.AuditCompletion.Contract) {
		t.Fatalf("target: %v %v", bindings, err)
	}
	if got := auditPinnedContextRef(run, contracts.ArtifactRef{Namespace: "inputs", Name: "task"}); !reflect.DeepEqual(got, run.AuditCompletion.Contract.Task) {
		t.Fatal("lost exact input")
	}
	ordinal := 1
	escalated, err := workflow.selectExecution(runstore.StageExecution{StageName: workflow.stageName, ExecutionConfigVariant: runstore.StageExecutionConfigFailedEscalation, EscalationOrdinal: &ordinal})
	if err != nil {
		t.Fatal(err)
	}
	escalatedBindings, err := auditBindingRequirements(run, escalated, pinned)
	if err != nil || !reflect.DeepEqual(escalatedBindings[0].CompletionContract, bindings[0].CompletionContract) {
		t.Fatal("escalation lost contract", err)
	}
	changed := exactRef("inputs", "task", "task-r2")
	pinned.Artifacts["task"] = runstore.PinnedContextArtifact{Required: true, Artifact: &changed}
	if _, err := auditBindingRequirements(run, workflow, pinned); err == nil {
		t.Fatal("accepted replaced task alias")
	}
	other, err := decodeExecutableWorkflow(h.store.run)
	if err != nil {
		t.Fatal(err)
	}
	other.stageName = "unselected-stage"
	otherBindings, err := auditBindingRequirements(run, other, runstore.StageContextSnapshot{})
	if err != nil || otherBindings[0].CompletionContract != nil {
		t.Fatal("non-targeted Worker was enrolled", err)
	}
	run.AuditCompletion.Agent = "another"
	if _, err := auditBindingRequirements(run, workflow, pinned); err == nil {
		t.Fatal("accepted another logical Worker")
	}
	run.AuditCompletion = nil
	run.PublicationMode = runstore.PublicationOrdinary
	run.MetadataLabels = runstore.RunMetadataLabels{"audit.id": "spoofed", "audit.role": "check"}
	if _, err := auditBindingRequirements(run, workflow, pinned); err == nil {
		t.Fatal("ordinary Run activated v2")
	}
	ordinary, err := decodeExecutableWorkflow(h.store.run)
	if err != nil {
		t.Fatal(err)
	}
	bindings, err = auditBindingRequirements(run, ordinary, runstore.StageContextSnapshot{})
	if err != nil || bindings[0].CompletionContract != nil {
		t.Fatal("labels activated ordinary Worker", err)
	}
}
