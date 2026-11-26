package streamline

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

type runtimeInvokerFunc func(context.Context, string, contracts.WorkerHandle, contracts.StageContentRequest) (contracts.WorkerCompletion, error)

func (f runtimeInvokerFunc) Invoke(ctx context.Context, name string, handle contracts.WorkerHandle, request contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
	return f(ctx, name, handle, request)
}

// Exercise the production Runtime host against the real session journal. A lost
// response replays an operation without repeating its durable transition or call.
func TestPostgresRuntimePlannerOrderedReplayAndRequiredEvent(t *testing.T) {
	for _, rejectFinish := range []bool{false, true} {
		name := "complete"
		if rejectFinish {
			name = "required-event-rejected"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
			defer cancel()
			pool := isolatedStreamlinePool(t, ctx)
			store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
			createPreparingStreamlineStage(t, ctx, store)
			if rejectFinish {
				installRequiredEventGuard(t, ctx, pool, "finish", "invalid")
			}
			sessions := migrationBoundarySession(t, store)
			workers := appendProbeWorkers()
			var factory *Factory
			invoker := runtimeInvokerFunc(func(callContext context.Context, name string, handle contracts.WorkerHandle, request contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
				if !handle.WorkerRuntimeRef.IsPlanner() {
					return workers.Invoke(callContext, name, handle, request)
				}
				if request.Planner == nil || request.Planner.Validate() != nil || request.Deadline.IsZero() {
					t.Fatal("Runtime invocation omitted its bounded Planner payload")
				}
				sequence := 0
				operation := func(name string, args any) contracts.PlannerOperationResult {
					t.Helper()
					sequence++
					encoded, err := json.Marshal(args)
					if err != nil {
						t.Fatal(err)
					}
					req := contracts.PlannerOperation{APIVersion: contracts.APIVersion, Sequence: sequence, Name: name, Arguments: encoded}
					if _, err := factory.ExecuteOperation(callContext, "stage-1", "wrong-allocation", req); err == nil {
						t.Fatal("another allocation entered the Planner host")
					}
					future := req
					future.Sequence++
					if _, err := factory.ExecuteOperation(callContext, "stage-1", handle.AllocationID, future); err == nil {
						t.Fatal("out-of-order request entered the Planner host")
					}
					reply, err := factory.ExecuteOperation(callContext, "stage-1", handle.AllocationID, req)
					if err != nil {
						t.Fatal(err)
					}
					replay, err := factory.ExecuteOperation(callContext, "stage-1", handle.AllocationID, req)
					first, _ := json.Marshal(reply)
					second, _ := json.Marshal(replay)
					if err != nil || string(first) != string(second) {
						t.Fatalf("lost response replay changed the outcome: %v", err)
					}
					return reply
				}
				for _, step := range []struct {
					name string
					args any
				}{
					{addSubtaskToolName, addSubtaskArgs{Objective: "Inspect bounded input", Instructions: "Return bounded evidence"}},
					{executeCurrentSubtaskToolName, workerCallArgs{SubtaskID: "0"}},
					{finishToolName, finishArgs{Outcome: stageresult.StageSucceeded, Summary: "Complete", Artifacts: map[string]materialSelection{}}},
				} {
					operation("before_model", map[string]any{"retainedBytes": 0})
					operation("after_model", map[string]any{"inputTokens": 7, "outputTokens": 3, "cachedTokens": 0, "calls": []string{step.name}, "cost": map[string]any{"currency": "USD", "calls": (sequence + 2) / 3, "groups": []any{map[string]any{"model": "fake-streamline-model", "phase": "main", "source": "gateway", "calls": (sequence + 2) / 3, "amount": fmt.Sprintf("0.0%d", (sequence+2)/3)}}}})
					reply := operation(step.name, step.args)
					if step.name == finishToolName && (reply.Failure != nil) != rejectFinish {
						t.Fatalf("finish failure = %+v, reject=%v", reply.Failure, rejectFinish)
					}
				}
				return workerCompletion("0", "Planner loop finished", map[string]contracts.ArtifactRef{}), nil
			})
			var err error
			factory, err = NewRuntimeFactoryWithMemory(sessions, invoker, &fakeInspector{}, unavailableWorkerStateReader{}, nil, Limits{MaxWallTime: time.Minute})
			if err != nil {
				t.Fatal(err)
			}
			invocation := appendProbeInvocation("streamline")
			handle := invocation.Workers["builder"].Clone()
			handle.AllocationID = "allocation-planner"
			handle.WorkerRuntimeRef = contracts.WorkerRuntimeRef{RuntimeID: "planner", Version: "1"}
			invocation.RuntimeExecutor = &handle
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			result, runErr := instance.Run(ctx)
			if rejectFinish {
				assertPlannerCode(t, runErr, "planner_session_rejected")
			} else if runErr != nil || result.Outcome != stageresult.StageSucceeded {
				t.Fatalf("Runtime candidate = %+v, %v", result, runErr)
			}
			if len(workers.bindingCalls()) != 1 {
				t.Fatal("replayed operation dispatched Worker more than once")
			}
			if _, ok := factory.runtimeHosts.Load("stage-1"); ok {
				t.Fatal("returned Planner retained an active operation host")
			}
			report, ok := instance.(planner.ReportProvider).ExecutionReport()
			if !ok || report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != 3 || report.Metrics.TotalTokens == nil || *report.Metrics.TotalTokens != 30 {
				t.Fatalf("replayed model operation charged usage twice: %+v", report.Metrics)
			}
			if cost := report.Metrics.LLMCost; cost == nil || cost.Calls != 3 || len(cost.Groups) != 1 || cost.Groups[0].Calls != 3 || cost.Groups[0].Amount != "0.03" {
				t.Fatalf("Runtime cost snapshot was lost/replayed: %+v", cost)
			}
		})
	}
}

func TestPostgresRuntimePlannerCancellationJoinsAdmittedWorker(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	pool := isolatedStreamlinePool(t, ctx)
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createPreparingStreamlineStage(t, ctx, store)
	sessions := migrationBoundarySession(t, store)
	started, stopped, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	callbackDone := make(chan struct{})
	var factory *Factory
	invoker := runtimeInvokerFunc(func(callContext context.Context, _ string, handle contracts.WorkerHandle, _ contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
		if !handle.WorkerRuntimeRef.IsPlanner() {
			close(started)
			<-callContext.Done()
			close(stopped)
			<-release
			return contracts.WorkerCompletion{}, callContext.Err()
		}
		request := contracts.PlannerOperation{APIVersion: contracts.APIVersion}
		for _, step := range []struct{ name, arguments string }{
			{"before_model", `{"retainedBytes":0}`},
			{"after_model", `{"inputTokens":1,"outputTokens":1,"cachedTokens":0,"calls":["add_subtask"]}`},
			{addSubtaskToolName, `{"objective":"Inspect input","instructions":"Return evidence"}`},
			{"before_model", `{"retainedBytes":0}`},
			{"after_model", `{"inputTokens":1,"outputTokens":1,"cachedTokens":0,"calls":["execute_current_subtask"]}`},
		} {
			request.Sequence++
			request.Name, request.Arguments = step.name, json.RawMessage(step.arguments)
			if reply, err := factory.ExecuteOperation(callContext, "stage-1", handle.AllocationID, request); err != nil || reply.Failure != nil {
				t.Error("could not initialize the plan")
				return contracts.WorkerCompletion{}, err
			}
		}
		go func() {
			defer close(callbackDone)
			request.Sequence, request.Name, request.Arguments = 6, executeCurrentSubtaskToolName, json.RawMessage(`{"subtask_id":"0"}`)
			_, _ = factory.ExecuteOperation(context.Background(), "stage-1", handle.AllocationID, request)
		}()
		<-callContext.Done()
		return contracts.WorkerCompletion{}, callContext.Err()
	})
	var err error
	factory, err = NewRuntimeFactoryWithMemory(sessions, invoker, &fakeInspector{}, unavailableWorkerStateReader{}, nil, Limits{MaxWallTime: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	invocation := appendProbeInvocation("streamline")
	handle := invocation.Workers["builder"].Clone()
	handle.AllocationID, handle.WorkerRuntimeRef = "allocation-planner", contracts.WorkerRuntimeRef{RuntimeID: "planner", Version: "1"}
	invocation.RuntimeExecutor = &handle
	instance, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := instance.Run(ctx); done <- err }()
	select {
	case <-started:
	case err := <-done:
		t.Fatalf("Planner ended before its Worker started: %v", err)
	}
	cancel()
	<-stopped
	select {
	case <-done:
		t.Fatal("Planner returned while its admitted Worker still ran")
	default:
	}
	close(release)
	assertPlannerCode(t, <-done, "planner_cancelled")
	<-callbackDone
	if _, ok := factory.runtimeHosts.Load("stage-1"); ok {
		t.Fatal("cancelled invocation retained its host")
	}
}

func TestPostgresRuntimePlannerClassifiedFailures(t *testing.T) {
	for _, test := range []struct {
		name, kind, code, want string
		retry                  bool
	}{
		{"null-admission", "invalid-admission", "", "planner_operation_invalid", false},
		{"missing-usage", "invalid_response", "", "planner_gateway_invalid_response", true},
		{"permanent-gateway", "rejected", "context_window_exceeded", "planner_gateway_rejected", false},
		{"gateway-unavailable", "unavailable", "", "planner_gateway_unavailable", true},
		{"model-budget", "", "", "planner_model_call_limit", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := t.Context()
			pool := isolatedStreamlinePool(t, ctx)
			store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
			createPreparingStreamlineStage(t, ctx, store)
			sessions := migrationBoundarySession(t, store)
			var factory *Factory
			invoker := runtimeInvokerFunc(func(call context.Context, _ string, handle contracts.WorkerHandle, _ contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
				sequence := 0
				op := func(name string, args any) contracts.PlannerOperationResult {
					t.Helper()
					sequence++
					encoded, _ := json.Marshal(args)
					r, err := factory.ExecuteOperation(call, "stage-1", handle.AllocationID, contracts.PlannerOperation{APIVersion: contracts.APIVersion, Sequence: sequence, Name: name, Arguments: encoded})
					if err != nil {
						t.Fatal(err)
					}
					return r
				}
				if test.kind == "invalid-admission" {
					result := op("before_model", map[string]any{"retainedBytes": nil})
					if result.Failure == nil || result.Failure.Code != test.want || result.Failure.Retryable {
						t.Fatalf("null admission escaped its contract: %+v", result.Failure)
					}
					return contracts.WorkerCompletion{}, nil
				}
				op("before_model", map[string]any{"retainedBytes": 0})
				var result contracts.PlannerOperationResult
				if test.kind == "" {
					op("after_model", map[string]any{"inputTokens": 1, "outputTokens": 1, "cachedTokens": 0, "calls": []string{}})
					result = op("before_model", map[string]any{"retainedBytes": 0})
				} else {
					result = op("model_failed", map[string]any{"kind": test.kind, "code": test.code, "usage": nil, "emptyStop": false, "outputLimited": false})
				}
				if result.Failure == nil || result.Failure.Code != test.want || result.Failure.Retryable != test.retry {
					t.Fatalf("operation failure = %+v", result.Failure)
				}
				return contracts.WorkerCompletion{}, nil
			})
			var err error
			factory, err = NewRuntimeFactoryWithMemory(sessions, invoker, &fakeInspector{}, unavailableWorkerStateReader{}, nil, Limits{MaxWallTime: time.Minute})
			if err != nil {
				t.Fatal(err)
			}
			invocation := appendProbeInvocation("streamline")
			invocation.ModelAccess.ModelPolicy.MaxModelCalls = 1
			handle := invocation.Workers["builder"].Clone()
			handle.AllocationID = "allocation-planner"
			handle.WorkerRuntimeRef = contracts.WorkerRuntimeRef{RuntimeID: "planner", Version: "1"}
			invocation.RuntimeExecutor = &handle
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			_, err = instance.Run(ctx)
			assertPlannerCode(t, err, test.want)
			if failure := planner.FailureFrom(err); failure.Retryable != test.retry {
				t.Fatalf("Run failure = %+v", failure)
			}
		})
	}
}

func TestPostgresRuntimePlannerCheckpointRequiresAcknowledgedMarker(t *testing.T) {
	for _, scenario := range []string{"accepted", "stale-generation", "lost-marker-ack"} {
		t.Run(scenario, func(t *testing.T) {
			ctx := t.Context()
			pool := isolatedStreamlinePool(t, ctx)
			store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
			createPreparingStreamlineStage(t, ctx, store)
			sessions := &markerProbeSessions{Service: migrationBoundarySession(t, store), lostAck: scenario == "lost-marker-ack"}
			var factory *Factory
			invoker := runtimeInvokerFunc(func(call context.Context, _ string, handle contracts.WorkerHandle, request contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
				if request.Planner == nil || request.Planner.Compaction == nil {
					t.Fatal("compaction configuration was lost")
				}
				sequence := 0
				op := func(name string, args any) contracts.PlannerOperationResult {
					t.Helper()
					sequence++
					encoded, _ := json.Marshal(args)
					req := contracts.PlannerOperation{APIVersion: contracts.APIVersion, Sequence: sequence, Name: name, Arguments: encoded}
					r, err := factory.ExecuteOperation(call, "stage-1", handle.AllocationID, req)
					if err != nil {
						t.Fatal(err)
					}
					// Lost HTTP reply must not repeat either usage or marker writes.
					if _, err := factory.ExecuteOperation(call, "stage-1", handle.AllocationID, req); err != nil {
						t.Fatal(err)
					}
					return r
				}
				op("before_compactor", map[string]any{"retainedBytes": 1000, "inputBytes": 1000, "messages": 2})
				if r := op("after_compactor", map[string]any{"inputTokens": 5, "outputTokens": 3, "cachedTokens": 0, "text": `{"summary":"Retain the completed work."}`, "finishReason": "stop", "toolCalls": 0, "sourceBytes": 1000}); r.Failure != nil {
					t.Fatalf("valid summary rejected: %+v", r.Failure)
				}
				generation := 1
				if scenario == "stale-generation" {
					generation++
				}
				r := op("checkpoint", map[string]any{"generation": generation, "retainedBytes": 500})
				if scenario == "accepted" {
					if r.Failure != nil {
						t.Fatal(r.Failure)
					}
				} else if r.Failure == nil {
					t.Fatal("unacknowledged checkpoint accepted")
				}
				return contracts.WorkerCompletion{}, nil
			})
			var err error
			factory, err = NewRuntimeFactoryWithMemory(sessions, invoker, compactionInspector(), unavailableWorkerStateReader{}, nil, Limits{MaxWallTime: time.Minute})
			if err != nil {
				t.Fatal(err)
			}
			invocation := compactionInvocation("streamline")
			handle := invocation.Workers["builder"].Clone()
			handle.AllocationID = "allocation-planner"
			handle.WorkerRuntimeRef = contracts.WorkerRuntimeRef{RuntimeID: "planner", Version: "1"}
			invocation.RuntimeExecutor = &handle
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			_, err = instance.Run(ctx)
			if scenario == "stale-generation" {
				assertPlannerCode(t, err, "planner_operation_invalid")
			}
			if scenario == "lost-marker-ack" {
				assertPlannerCode(t, err, "planner_session_unavailable")
			}
			want := 1
			if scenario == "stale-generation" {
				want = 0
			}
			if got := sessions.count(); got != want {
				t.Fatalf("marker appends = %d, want %d", got, want)
			}
			report, _ := instance.(planner.ReportProvider).ExecutionReport()
			if report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != 1 || report.Metrics.TotalTokens == nil || *report.Metrics.TotalTokens != 8 {
				t.Fatalf("compactor replay changed usage: %+v", report.Metrics)
			}
		})
	}
}
