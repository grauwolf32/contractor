package scheduler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The two participants bound their model requests differently, so each
// probe lease must follow its holder's bound.
const (
	probeTestWorkerRequestTimeout  = time.Minute
	probeTestPlannerRequestTimeout = 30 * time.Minute
)

func newProbeTestRecovery(t *testing.T, pool *pgxpool.Pool) *gatewayrecovery.Service {
	t.Helper()
	service, err := gatewayrecovery.New(
		pool, runstore.GatewayRecoveryRuns{}, gatewayrecovery.DefaultPolicy(),
		probeTestWorkerRequestTimeout, probeTestPlannerRequestTimeout,
	)
	if err != nil {
		t.Fatal(err)
	}
	return service
}

// admitRunOnRoute creates a running Run admitted on route and returns its
// planner participant.
func admitRunOnRoute(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, service *gatewayrecovery.Service,
	runID string, route gatewayrecovery.Route,
) *gatewayrecovery.Participant {
	t.Helper()
	runs := runstore.NewPostgresStore(pool)
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel,
		RunID:     runID, OwnerID: "user-1", WorkflowName: "test", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
		RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	if allowed, err := service.Admit(ctx, runID, []gatewayrecovery.Route{route}); err != nil || !allowed {
		t.Fatalf("admit Run %s = %v, %v", runID, allowed, err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	if err := runstoretest.CreateRunningStage(ctx, runs, runID, "stage-"+runID, contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	return service.Planner(runID, "stage-"+runID, route, llmgateway.DefaultGatewayFailureSignatures())
}

// blockRouteForOtherRun admits a second running Run on route, whose planner
// reports an outage that blocks the route, and makes a probe due.
func blockRouteForOtherRun(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, service *gatewayrecovery.Service, route gatewayrecovery.Route,
) *gatewayrecovery.Participant {
	t.Helper()
	other := admitRunOnRoute(t, ctx, pool, service, "run-other", route)
	recoveryUpdate(t, other, "other-outage", "failed")
	if _, err := pool.Exec(ctx, `UPDATE gateway_recovery_routes SET next_probe_at=clock_timestamp()-interval '1 second'`); err != nil {
		t.Fatal(err)
	}
	return other
}

// requireProbeLease checks that holderRunID holds the route's probe with a
// lease sized by its own request timeout.
func requireProbeLease(t *testing.T, ctx context.Context, pool *pgxpool.Pool, route gatewayrecovery.Route, holderRunID string, requestTimeout time.Duration) {
	t.Helper()
	var holder *string
	var lease time.Duration
	var remaining float64
	if err := pool.QueryRow(ctx, `
SELECT probe_run_id, EXTRACT(EPOCH FROM probe_until-clock_timestamp())
FROM gateway_recovery_routes WHERE route_key=$1`, route.Key()).Scan(&holder, &remaining); err != nil {
		t.Fatal(err)
	}
	lease = time.Duration(remaining * float64(time.Second))
	want := requestTimeout + gatewayrecovery.DefaultPolicy().InitialDelay
	if holder == nil || *holder != holderRunID || lease > want || lease < want-time.Minute/2 {
		t.Fatalf("probe holder=%q lease=%s, want %s for about %s", stringValue(holder), lease, holderRunID, want)
	}
}

// requireProbeFreed checks that the other Run probes the still blocked route
// at once, without waiting for the holder's lease, and that the holder no
// longer waits. stageEnded also requires the holder's route membership gone.
func requireProbeFreed(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, other *gatewayrecovery.Participant,
	route gatewayrecovery.Route, holderRunID string, stageEnded bool,
) {
	t.Helper()
	if decision := recoveryUpdate(t, other, "other-probe", "acquire"); !decision.Allowed {
		t.Fatalf("other Run could not probe after the holder's Stage ended: %+v", decision)
	}
	var blocked bool
	var failures int64
	var probeRun *string
	if err := pool.QueryRow(ctx, `
SELECT blocked, failure_count, probe_run_id FROM gateway_recovery_routes WHERE route_key=$1`, route.Key()).Scan(
		&blocked, &failures, &probeRun,
	); err != nil {
		t.Fatal(err)
	}
	if !blocked || failures != 1 || probeRun == nil || *probeRun != "run-other" {
		t.Fatalf("route after Stage end: blocked=%v failures=%d probe Run=%q", blocked, failures, stringValue(probeRun))
	}
	var waits, memberships int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM gateway_recovery_waits WHERE run_id=$1),
       (SELECT count(*) FROM gateway_run_routes WHERE run_id=$1)`, holderRunID).Scan(&waits, &memberships); err != nil {
		t.Fatal(err)
	}
	if waits != 0 || (stageEnded && memberships != 0) {
		t.Fatalf("holder kept %d waits and %d route memberships", waits, memberships)
	}
}

// A Worker holds the probe of a blocked route when its Stage is aborted and
// never reports the end of its call, as when its release meets the write
// fence. The abort's terminal commit relinquishes the probe, so another Run on
// the route probes at once rather than after the Worker's lease (SCH-R108).
func TestPostgresAbortedStageReleasesWorkerProbe(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	h := newDeferredPlacementHarness(t, ctx, nil)
	service := newProbeTestRecovery(t, h.pool)
	h.scheduler.options.GatewayRecovery = service
	stage := h.workflow.Stages[h.workflow.EntryStage]
	route := reservationModelRoutes("user-1", schedulerTestReservations(t, stage)[0])[0]
	var other *gatewayrecovery.Participant
	h.invoker.before = func(ctx context.Context, handle contracts.WorkerHandle) error {
		other = blockRouteForOtherRun(t, ctx, h.pool, service, route)
		decision, err := service.Update(ctx, handle.AllocationID, gatewayrecovery.Request{
			Model: route.Model, RequestID: "worker-probe", Action: "acquire",
		})
		if err != nil || !decision.Allowed {
			t.Fatalf("Worker probe = %+v, %v", decision, err)
		}
		requireProbeLease(t, ctx, h.pool, route, "run-1", probeTestWorkerRequestTimeout)
		if decision := recoveryUpdate(t, other, "other-waiting", "acquire"); decision.Allowed {
			t.Fatal("other Run acquired the probe the Worker holds")
		}
		return planner.NewError("worker_unavailable", "Worker lost its model", true, errors.New("model call aborted"))
	}

	if worked, err := h.scheduler.RunOnce(ctx); !worked || err != nil {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	executions, err := h.store.ListStageExecutions(ctx, "run-1")
	if err != nil || len(executions) != 1 || executions[0].State != runstore.StageInterrupted {
		t.Fatalf("aborted Stage = (%+v, %v)", executions, err)
	}
	if other == nil {
		t.Fatal("Worker never held the probe")
	}
	requireProbeFreed(t, ctx, h.pool, other, route, "run-1", true)
}

// Every transaction that ends a Stage relinquishes the probe its Run holds;
// each starts from a planner holding the probe while its Run waits.
func TestPostgresEveryStageEndRelinquishesProbe(t *testing.T) {
	enterFinalizing := func(t *testing.T, ctx context.Context, f finalizingFixture) {
		t.Helper()
		if err := f.persistence.EnterFinalizingWithResult(ctx, runstore.EnterFinalizingParams{
			StageExecutionID: f.executionID, ResultSchemaVersion: contracts.APIVersion,
			Candidate: f.result, FinalizationID: "finalization-probe", Deadline: time.Now().Add(time.Minute),
			Reason: runstore.Reason{Code: "planner_completed"},
		}); err != nil {
			t.Fatal(err)
		}
	}
	enterAborting := func(t *testing.T, ctx context.Context, f finalizingFixture) {
		t.Helper()
		if err := f.persistence.EnterAbortingWithTermination(ctx, "run-1", runstore.EnterAbortingParams{
			StageExecutionID: f.executionID, ExpectedState: runstore.StageRunning,
			TerminationSchemaVersion: contracts.APIVersion,
			Termination: runstore.StageTermination{
				Outcome: runstore.TerminationInterrupted, Code: "worker_unavailable", Message: "Worker lost",
				Retryable: true, Phase: runstore.TerminationRunning, OccurredAt: time.Now().UTC(),
			},
			AbortID: "abort-probe", Deadline: time.Now().Add(time.Minute),
			Reason: runstore.Reason{Code: "stage_interrupted"},
		}); err != nil {
			t.Fatal(err)
		}
	}
	cancelRun := func(t *testing.T, ctx context.Context, f finalizingFixture) {
		t.Helper()
		if _, err := f.store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{
			Code: runstore.CancellationUserRequested, RequestedAt: time.Now(),
		}); err != nil {
			t.Fatal(err)
		}
	}
	for _, test := range []struct {
		name string
		end  func(*testing.T, context.Context, finalizingFixture) error
		want runstore.WorkflowRunState
	}{
		{"accepted result", func(t *testing.T, ctx context.Context, f finalizingFixture) error {
			enterFinalizing(t, ctx, f)
			return f.persistence.CommitResultProgression(ctx, ResultProgression{
				RunID: "run-1", StageExecutionID: f.executionID, Result: f.result,
				WorkflowOutputs: f.workflow.Stages["copy"].WorkflowOutputs, OutputContracts: f.workflow.Outputs,
				Progression: terminalSuccessProgression("run-1", f.executionID),
			})
		}, runstore.RunSucceeded},
		{"result accepted during cancellation", func(t *testing.T, ctx context.Context, f finalizingFixture) error {
			enterFinalizing(t, ctx, f)
			cancelRun(t, ctx, f)
			return f.persistence.AcceptResultDuringCancellation(ctx, "run-1", f.executionID, f.result)
		}, runstore.RunCancelled},
		{"termination progression", func(t *testing.T, ctx context.Context, f finalizingFixture) error {
			enterAborting(t, ctx, f)
			return f.persistence.CommitTerminationProgression(ctx, TerminationProgression{
				RunID: "run-1", StageExecutionID: f.executionID,
				Progression: StageProgression{
					Decision: runstore.RecordStageTransitionDecisionParams{
						SourceExecutionID: f.executionID, RunID: "run-1", Action: runstore.StageTransitionFail,
					},
					TerminalRunState: runstore.RunFailed, RunReason: runstore.Reason{Code: "worker_unavailable"},
				},
			})
		}, runstore.RunFailed},
		{"termination during cancellation", func(t *testing.T, ctx context.Context, f finalizingFixture) error {
			cancelRun(t, ctx, f)
			enterAborting(t, ctx, f)
			return f.persistence.CommitTerminationAndFinishRun(
				ctx, "run-1", f.executionID, runstore.RunCancelling, runstore.RunCancelled,
				runstore.Reason{Code: runstore.CancellationUserRequested},
			)
		}, runstore.RunCancelled},
		{"unprogressable waiting Run", func(_ *testing.T, ctx context.Context, f finalizingFixture) error {
			return f.persistence.FailRunWithActiveStages(
				ctx, "run-1", runstore.RunWaiting, runstore.RunFailed,
				runstore.Reason{Code: "scheduler_state_invalid"},
				runstore.StageTermination{
					Outcome: runstore.TerminationInterrupted, Code: "scheduler_state_invalid",
					Message: "Scheduler could not progress the WorkflowRun", OccurredAt: time.Now().UTC(),
				},
				"abort-invalid",
			)
		}, runstore.RunFailed},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool := isolatedSchedulerPool(t, ctx)
			service := newProbeTestRecovery(t, pool)
			f := createRunningStageFixture(t, ctx, pool)
			route := gatewayrecovery.Route{OwnerID: "user-1", GatewayDigest: "gateway", Model: "model"}
			if allowed, err := service.Admit(ctx, "run-1", []gatewayrecovery.Route{route}); err != nil || !allowed {
				t.Fatalf("admit holder Run = %v, %v", allowed, err)
			}
			holder := service.Planner("run-1", f.executionID, route, llmgateway.DefaultGatewayFailureSignatures())
			other := blockRouteForOtherRun(t, ctx, pool, service, route)
			if !recoveryUpdate(t, holder, "holder-probe", "acquire").Allowed {
				t.Fatal("holder did not acquire the due probe")
			}
			requireProbeLease(t, ctx, pool, route, "run-1", probeTestPlannerRequestTimeout)
			if run, err := f.store.GetRun(ctx, "run-1"); err != nil || run.State != runstore.RunWaiting {
				t.Fatalf("probe holder Run = (%s, %v), want waiting", run.State, err)
			}

			if err := test.end(t, ctx, f); err != nil {
				t.Fatalf("end Stage: %v", err)
			}
			if run, err := f.store.GetRun(ctx, "run-1"); err != nil || run.State != test.want {
				t.Fatalf("holder Run after Stage end = (%s, %v), want %s", run.State, err, test.want)
			}
			requireProbeFreed(t, ctx, pool, other, route, "run-1", true)
		})
	}
}

func stringValue(value *string) string {
	if value == nil {
		return "<none>"
	}
	return *value
}

// A Server-side planner request cancelled while it holds the probe releases
// it on a detached context and reports the cancellation, not a Gateway
// failure: the route keeps its failure count and another Run probes at once.
func TestPostgresCancelledPlannerRequestReleasesProbe(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	service := newProbeTestRecovery(t, pool)
	route := gatewayrecovery.Route{OwnerID: "user-1", GatewayDigest: "gateway", Model: "model"}
	holder := admitRunOnRoute(t, ctx, pool, service, "run-planner", route)
	other := blockRouteForOtherRun(t, ctx, pool, service, route)
	received, unblock := make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		close(received)
		// A model that has not answered when the invocation is cancelled.
		<-unblock
	}))
	t.Cleanup(server.Close)
	t.Cleanup(func() { close(unblock) })

	requestContext, cancelRequest := context.WithCancel(ctx)
	defer cancelRequest()
	request, err := http.NewRequestWithContext(
		requestContext, http.MethodPost, server.URL+"/v1/chat/completions", bytes.NewReader([]byte(`{}`)),
	)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, err := holder.Do(request, server.Client(), 1<<20)
		done <- err
	}()
	select {
	case <-received:
	case err := <-done:
		t.Fatalf("planner request ended before reaching the Gateway: %v", err)
	}
	requireProbeLease(t, ctx, pool, route, "run-planner", probeTestPlannerRequestTimeout)
	cancelRequest()
	err = <-done
	var gatewayFailure *gatewayrecovery.FailureError
	if !errors.Is(err, context.Canceled) || errors.As(err, &gatewayFailure) {
		t.Fatalf("cancelled planner request = %v, want the cancellation", err)
	}
	var failureReceipts int
	var state runstore.WorkflowRunState
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM gateway_recovery_failures WHERE route_key=$1),
       (SELECT state FROM workflow_runs WHERE run_id='run-planner')`, route.Key()).Scan(&failureReceipts, &state); err != nil {
		t.Fatal(err)
	}
	if failureReceipts != 1 || state != runstore.RunRunning {
		t.Fatalf("failure receipts=%d (want only the other Run's outage), holder Run %s (want running)", failureReceipts, state)
	}
	requireProbeFreed(t, ctx, pool, other, route, "run-planner", false)
}
