package streamline

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

func TestPostgresRuntimeRouterDispatchHistoryAndReplay(t *testing.T) {
	ctx := t.Context()
	pool := isolatedStreamlinePool(t, ctx)
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createPreparingStreamlineStage(t, ctx, store)
	sessions := migrationBoundarySession(t, store)
	workers := &fakeWorkerInvoker{results: map[string]stageresult.StageContentResult{
		"builder":  stageResult(strings.Repeat("builder evidence ", 3000), map[string]contracts.ArtifactRef{}),
		"reviewer": stageResult("reviewer evidence", map[string]contracts.ArtifactRef{}),
	}}
	var factory *Factory
	var registry *planner.Registry
	invoker := runtimeInvokerFunc(func(call context.Context, name string, h contracts.WorkerHandle, r contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
		if r.Planner == nil {
			return workers.Invoke(call, name, h, r)
		}
		seq := 0
		op := func(name string, args any) contracts.PlannerOperationResult {
			t.Helper()
			seq++
			data, _ := json.Marshal(args)
			request := contracts.PlannerOperation{APIVersion: contracts.APIVersion, Sequence: seq, Name: name, Arguments: data}
			reply, err := registry.ExecuteOperation(call, "stage-1", h.AllocationID, request)
			if err != nil {
				t.Fatal(err)
			}
			replay, e := registry.ExecuteOperation(call, "stage-1", h.AllocationID, request)
			a, _ := json.Marshal(reply)
			b, _ := json.Marshal(replay)
			if e != nil || string(a) != string(b) {
				t.Fatal("ordered reply changed", e)
			}
			if reply.Failure != nil {
				t.Fatalf("%s: %+v", name, reply.Failure)
			}
			return reply
		}
		tool := func(name string, args any) map[string]any {
			t.Helper()
			op("before_model", map[string]any{"retainedBytes": 0})
			op("after_model", map[string]any{"inputTokens": 1, "outputTokens": 1, "cachedTokens": 0, "calls": []string{name}})
			reply := op(name, args)
			var out map[string]any
			if err := json.Unmarshal(reply.Result, &out); err != nil {
				t.Fatal(err)
			}
			return out
		}
		tool(addSubtaskToolName, map[string]any{"objective": "Build", "instructions": "Inspect the source"})
		rejected := tool(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0", "worker_name": "foreign"})
		if rejected["ok"] == true || len(workers.bindingCalls()) != 0 {
			t.Fatal("unknown Worker dispatched")
		}
		tool(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0", "worker_name": "builder"})
		tool(addSubtaskToolName, map[string]any{"objective": "Review", "instructions": "Review the evidence"})
		tool(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "1", "worker_name": "reviewer"})
		first := tool(readContextHistoryToolName, map[string]any{"worker_name": "builder"})
		cursor, ok := first["next"].(string)
		if !ok || cursor == "" {
			t.Fatal("long history did not page", first)
		}
		foreign := tool(readContextHistoryToolName, map[string]any{"worker_name": "reviewer", "cursor": cursor})
		data, _ := json.Marshal(foreign)
		if !strings.Contains(string(data), "history_cursor_invalid") || strings.Contains(string(data), "builder evidence") {
			t.Fatal("Worker cursor scope leaked", foreign)
		}
		review := tool(readContextHistoryToolName, map[string]any{"worker_name": "reviewer"})
		data, _ = json.Marshal(review)
		if !strings.Contains(string(data), "reviewer evidence") || strings.Contains(string(data), "builder evidence") {
			t.Fatal("wrong Worker history", review)
		}
		op("before_compactor", map[string]any{"retainedBytes": 0, "inputBytes": 1000, "messages": 2})
		op("after_compactor", map[string]any{"inputTokens": 1, "outputTokens": 1, "cachedTokens": 0, "text": `{"summary":"Both subtasks completed."}`, "finishReason": "stop", "toolCalls": 0, "sourceBytes": 1000})
		op("checkpoint", map[string]any{"generation": 1, "retainedBytes": 0})
		stale := tool(readContextHistoryToolName, map[string]any{"worker_name": "builder", "cursor": cursor})
		data, _ = json.Marshal(stale)
		if !strings.Contains(string(data), "history_cursor_invalid") {
			t.Fatal("checkpoint retained stale cursor", stale)
		}
		tool(finishToolName, map[string]any{"outcome": "succeeded", "summary": "Complete", "artifacts": map[string]any{}})
		return workerCompletion("0", "Planner completed", nil), nil
	})
	var err error
	factory, err = NewRuntimeFactoryWithMemory(sessions, invoker, &fakeInspector{}, unavailableWorkerStateReader{}, nil, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	registry, err = planner.NewRegistry(factory)
	if err != nil {
		t.Fatal(err)
	}
	invocation := testInvocation("builder", "reviewer")
	settings := compactionInvocation("router")
	invocation.Stage.Planner = settings.Stage.Planner
	invocation.Stage.PlannerContextManagement = settings.Stage.PlannerContextManagement
	invocation.Stage.ExecutionConfig.Planner = settings.Stage.ExecutionConfig.Planner
	invocation.ModelAccess.ModelPolicy = settings.ModelAccess.ModelPolicy
	invocation.Stage.Result.Artifacts = map[string]resolved.ArtifactSlot{}
	instance, err := factory.CreateRouter(invocation)
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(ctx)
	if err != nil || result.Outcome != stageresult.StageSucceeded {
		t.Fatalf("Run: %+v %v", result, err)
	}
	if calls := workers.bindingCalls(); len(calls) != 2 || calls[0] != "builder" || calls[1] != "reviewer" {
		t.Fatal("dispatch duplicated or misrouted", calls)
	}
}
