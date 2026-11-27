package streamline

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/planner/stateview"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

func TestRuntimePlannerAndWorkerShareLogicalMemoryNamespace(t *testing.T) {
	store := newPlannerMemoryArtifactStore()
	invocation := testInvocation("builder")
	binding := invocation.Stage.Agents["builder"]
	binding.Template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "memory-tools", Version: "1"}, Tools: plannermemory.OperationNames()}}
	invocation.Stage.Agents["builder"] = binding
	factory, err := NewRuntimeFactoryWithMemory(newFakeSessions(), &fakeWorkerInvoker{}, &fakeInspector{}, unavailableWorkerStateReader{}, store, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	current, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	p := current.(*streamlinePlanner)
	defer p.stateViews.Close()
	state := newExecutionState(p.limits)
	tool := &plannerMemoryTool{planner: p, state: state, operation: plannermemory.ToolWriteMemory, bindings: p.memoryBindings(plannermemory.ToolWriteMemory)}
	result, err := tool.run(t.Context(), map[string]any{"name": "shared_note", "content": "Planner observation", "description": "Shared note", "tags": []any{"test"}})
	if err != nil || result["error"] != nil {
		t.Fatalf("write: %v %v", result, err)
	}
	worker, err := plannermemory.NewNamespace(store, plannermemory.Binding{RunID: invocation.RunID, StageExecutionID: invocation.StageExecutionID, Namespace: binding.Namespace})
	if err != nil {
		t.Fatal(err)
	}
	note, err := worker.ReadMemory(t.Context(), "shared_note")
	if err != nil || note.Content != "Planner observation" {
		t.Fatalf("Worker read: %+v %v", note, err)
	}
	if _, err := worker.AppendMemory(t.Context(), "shared_note", "Worker observation"); err != nil {
		t.Fatal(err)
	}
	tool.operation = plannermemory.ToolReadMemory
	tool.bindings = p.memoryBindings(tool.operation)
	result, err = tool.run(t.Context(), map[string]any{"name": "shared_note"})
	if err != nil || result["content"] != "Planner observation\nWorker observation" {
		t.Fatalf("Planner read: %v %v", result, err)
	}
}

func TestPostgresRuntimeAdvisoryStateFailurePreservesCompletedSubtask(t *testing.T) {
	pool := isolatedStreamlinePool(t, t.Context())
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createPreparingStreamlineStage(t, t.Context(), store)
	sessions := migrationBoundarySession(t, store)
	workers := appendProbeWorkers()
	script := &operationScript{steps: []operationStep{
		addSubtaskStep("Inspect", "Return evidence"),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(getWorkerToolUsageToolName, map[string]any{}),
		functionStep(finishToolName, map[string]any{"outcome": "succeeded", "summary": "Complete", "artifacts": map[string]any{}}),
	}}
	factory, err := runtimeScriptFactory(streamlineProfile, sessions, workers, &fakeInspector{}, unavailableWorkerStateReader{}, script, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	instance, err := factory.Create(appendProbeInvocation("streamline"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != stageresult.StageSucceeded {
		t.Fatalf("advisory read changed outcome: %+v %v", result, err)
	}
	if len(workers.bindingCalls()) != 1 {
		t.Fatal("advisory failure repeated Worker work")
	}
	recovered, err := migrationBoundarySession(t, store).Begin(t.Context(), "stage-1")
	if err != nil || recovered.Invoke || recovered.Completion == nil || recovered.Completion.Result == nil || recovered.Completion.Result.Outcome != stageresult.StageSucceeded {
		t.Fatalf("durable completion changed: %+v %v", recovered, err)
	}
	report, _ := instance.(*streamlinePlanner).ExecutionReport()
	encoded, err := json.Marshal(report)
	if err != nil || !strings.Contains(string(encoded), stateview.CodeUnavailable) {
		t.Fatalf("State failure not exercised: %s %v", encoded, err)
	}
}

func TestRuntimeStateProjectionFailureUsesClosedCodes(t *testing.T) {
	for _, tc := range []struct {
		input     string
		retry     bool
		want      string
		wantRetry bool
	}{
		{stateview.CodeUnavailable, false, stateview.CodeUnavailable, false},
		{stateview.CodeChanged, false, stateview.CodeChanged, true},
		{stateview.CodeWorkspaceIncomplete, true, stateview.CodeWorkspaceIncomplete, false},
		{"private_runtime_secret", false, stateview.CodeUnavailable, true},
	} {
		failure := stateToolFailure(&stateview.Error{Code: tc.input, Retryable: tc.retry})
		if failure.Code != tc.want || failure.Retryable != tc.wantRetry || strings.Contains(failure.Message, "private_runtime_secret") {
			t.Fatalf("unclassified failure: %+v", failure)
		}
	}
}

func TestPostgresRuntimeActivityPersistsOnlyContentFreeFacts(t *testing.T) {
	const canary = "private-content-cursor-provider-memory-canary"
	pool := isolatedStreamlinePool(t, t.Context())
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createPreparingStreamlineStage(t, t.Context(), store)
	script := &operationScript{content: canary, steps: []operationStep{
		functionStep(plannermemory.ToolWriteMemory, map[string]any{"name": "shared_note", "content": canary, "description": canary, "tags": []any{"private-tag-canary"}}),
		addSubtaskStep("Inspect", "Return evidence"),
		functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
		functionStep(getWorkerToolUsageToolName, map[string]any{"cursor": canary, "path": canary}),
		functionStep(finishToolName, map[string]any{"outcome": "succeeded", "summary": "Complete", "artifacts": map[string]any{}}),
	}}
	factory, err := runtimeScriptFactory(streamlineProfile, migrationBoundarySession(t, store), appendProbeWorkers(), &fakeInspector{}, unavailableWorkerStateReader{}, script, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	factory.memoryStore = newPlannerMemoryArtifactStore()
	invocation := appendProbeInvocation("streamline")
	invocation.ModelAccess.ModelPolicy.MaxModelCalls = 8
	binding := invocation.Stage.Agents["builder"]
	binding.Template.Toolsets = append(binding.Template.Toolsets, contracts.ToolsetSelection{Ref: contracts.ToolsetRef{ToolsetID: "memory-tools", Version: "1"}, Tools: []string{plannermemory.ToolWriteMemory}})
	invocation.Stage.Agents["builder"] = binding
	instance, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil || result.Outcome != stageresult.StageSucceeded {
		t.Fatalf("Run: %+v %v", result, err)
	}
	page, err := store.ListRunEvents(t.Context(), invocation.RunID, 0, 100)
	if err != nil || page.More {
		t.Fatalf("events: %v", err)
	}
	count := 0
	for _, event := range page.Events {
		if strings.Contains(string(event.Data), canary) || strings.Contains(string(event.Data), "private-tag-canary") {
			t.Fatal("private content retained in durable event")
		}
		if strings.Contains(string(event.Data), `"functionCalls"`) {
			count++
		}
	}
	if count < 10 {
		t.Fatalf("activity path not exercised: %d", count)
	}
}
