package scheduler

import (
	"context"
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// The delayed update is delivered only after the Scheduler's terminal commit
// and the next Planner start, fixing the race order without wall-clock sleeps.
func TestPostgresRecoveryRefusesLateEndedStageUpdates(t *testing.T) {
	for _, participantKind := range []string{"planner", "allocation"} {
		for _, action := range []string{"failed", "blocked-acquire", "healthy-acquire"} {
			t.Run(participantKind+"/"+action, func(t *testing.T) {
				ctx := t.Context()
				pool := isolatedSchedulerPool(t, ctx)
				service := newProbeTestRecovery(t, pool)
				f := createRunningStageFixture(t, ctx, pool)
				route := gatewayrecovery.Route{OwnerID: "user-1", GatewayDigest: "gateway", Model: "model"}
				if allowed, err := service.Admit(ctx, "run-1", []gatewayrecovery.Route{route}); err != nil || !allowed {
					t.Fatalf("admit = %v, %v", allowed, err)
				}
				old := service.Planner("run-1", f.executionID, route, llmgateway.DefaultGatewayFailureSignatures())
				update := old.Update
				if participantKind == "allocation" {
					h := newSchedulerHarness(t)
					reservation, err := h.allocator.reservation(f.executionID)
					if err != nil {
						t.Fatal(err)
					}
					allocation := stageAllocationFromReservation(reservation)
					if err := f.store.RecordStageAllocation(ctx, allocation); err != nil {
						t.Fatal(err)
					}
					if err := service.Bind(ctx, allocation.AllocationID, "run-1", route); err != nil {
						t.Fatal(err)
					}
					update = func(ctx context.Context, request gatewayrecovery.Request) (gatewayrecovery.Decision, error) {
						request.Model = route.Model
						return service.Update(ctx, allocation.AllocationID, request)
					}
				}
				if action == "blocked-acquire" {
					if _, err := update(ctx, gatewayrecovery.Request{RequestID: "old-outage", Action: "failed", Code: "gateway_unavailable"}); err != nil {
						t.Fatal(err)
					}
				}
				nextID := advanceRecoveryStage(t, ctx, f)
				if participantKind == "allocation" {
					// A restarted authority must reject the same durable old binding.
					service = newProbeTestRecovery(t, pool)
				}
				request := gatewayrecovery.Request{RequestID: "late-call", Action: "acquire"}
				if action == "failed" {
					request.Action, request.Code = "failed", "gateway_unavailable"
				}
				decision, err := update(ctx, request)
				if !gatewayrecovery.IsUnavailable(err) || decision.Allowed {
					t.Errorf("ended Stage update = %+v, %v, want unavailable", decision, err)
				}
				run, err := f.store.GetRun(ctx, "run-1")
				if err != nil || run.State != runstore.RunRunning {
					t.Errorf("next Stage Run = %s, %v, want running", run.State, err)
				}
				var waits, receipts int
				if err := pool.QueryRow(ctx, `SELECT
 (SELECT count(*) FROM gateway_recovery_waits WHERE run_id='run-1'),
 (SELECT count(*) FROM gateway_recovery_failures WHERE request_id='late-call')`).Scan(&waits, &receipts); err != nil {
					t.Fatal(err)
				}
				if waits != 0 || receipts != 0 {
					t.Errorf("ended Stage created waits=%d, receipts=%d", waits, receipts)
				}
				for _, terminal := range []string{"succeeded", "finished", "released"} {
					if _, err := update(ctx, gatewayrecovery.Request{RequestID: "old-call", Action: terminal}); err != nil {
						t.Errorf("ended Stage %s: %v", terminal, err)
					}
				}
				currentRoute := route
				currentRoute.Model = "current-model"
				if allowed, err := service.Admit(ctx, "run-1", []gatewayrecovery.Route{currentRoute}); err != nil || !allowed {
					t.Fatalf("admit next Stage route = %v, %v", allowed, err)
				}
				current := service.Planner("run-1", nextID, currentRoute, llmgateway.DefaultGatewayFailureSignatures())
				if _, err := current.Update(ctx, gatewayrecovery.Request{RequestID: "current-outage", Action: "failed", Code: "gateway_unavailable"}); err != nil {
					t.Fatalf("active Stage lost authority: %v", err)
				}
			})
		}
	}
}

func advanceRecoveryStage(t *testing.T, ctx context.Context, f finalizingFixture) string {
	t.Helper()
	previous, err := f.store.GetStageExecution(ctx, f.executionID)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.persistence.EnterFinalizingWithResult(ctx, runstore.EnterFinalizingParams{
		StageExecutionID: f.executionID, ResultSchemaVersion: contracts.APIVersion,
		Candidate: f.result, FinalizationID: "finalize-old", Deadline: time.Now().Add(time.Minute),
		Reason: runstore.Reason{Code: "planner_completed"},
	}); err != nil {
		t.Fatal(err)
	}
	const nextID = "stage-next"
	stageName := previous.StageName
	var pins []ContextPin
	for name, input := range previous.StageContext.Artifacts {
		if input.Artifact != nil {
			pins = append(pins, ContextPin{Name: name, Ref: *input.Artifact})
		}
	}
	if err := f.persistence.CommitResultProgression(ctx, ResultProgression{
		RunID: "run-1", StageExecutionID: f.executionID, Result: f.result,
		WorkflowOutputs: f.workflow.Stages[stageName].WorkflowOutputs, OutputContracts: f.workflow.Outputs,
		Progression: StageProgression{
			Decision: runstore.RecordStageTransitionDecisionParams{
				SourceExecutionID: f.executionID, RunID: "run-1", Action: runstore.StageTransitionNext,
				TargetStageName: &stageName, TargetExecutionID: stringPointer(nextID),
			},
			NextStage: &NextStageCreation{Params: runstore.CreateStageExecutionParams{
				LaneClass:        resolved.LaneClassModel,
				StageExecutionID: nextID, RunID: "run-1", StageName: stageName, Attempt: 2, PreviousExecutionID: &f.executionID,
				ExecutionConfigVariant: runstore.StageExecutionConfigBase,
				StageSpecSchemaVersion: previous.StageSpecSchemaVersion, StageSpecSnapshot: previous.StageSpecSnapshot,
				StageContextSchemaVersion: previous.StageContextSchemaVersion, StageContext: previous.StageContext,
			}, ContextPins: pins},
		},
	}); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(map[string]string{"stageExecutionId": nextID, "sessionId": "next-session", "invocationId": "next-invocation"})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.store.StartPlanner(ctx, runstore.StartPlannerParams{
		StageExecutionID: nextID, SessionID: "next-session", InvocationID: "next-invocation",
		StateSchemaVersion: contracts.APIVersion, InitialState: json.RawMessage(`{"status":"test"}`),
		RunEvent: runstore.RunEventAppend{EventID: "next-started", EventSchemaVersion: contracts.APIVersion, Kind: runstore.RunEventPlannerStarted, Data: data},
		Reason:   runstore.Reason{Code: "planner_started"},
	}); err != nil {
		t.Fatal(err)
	}
	return nextID
}
