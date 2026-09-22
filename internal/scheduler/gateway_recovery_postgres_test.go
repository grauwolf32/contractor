package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgxpool"
)

type recoveryFixture struct {
	service *gatewayrecovery.Service
	pool    *pgxpool.Pool
	runs    *runstore.PostgresStore
	route   gatewayrecovery.Route
}

func newRecoveryFixture(t *testing.T) recoveryFixture {
	t.Helper()
	pool := isolatedSchedulerPool(t, t.Context())
	service, err := gatewayrecovery.New(pool, gatewayrecovery.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	return recoveryFixture{service, pool, runstore.NewPostgresStore(pool), gatewayrecovery.Route{OwnerID: "owner", GatewayDigest: "gateway", Model: "model"}}
}
func (f recoveryFixture) run(t *testing.T, id string, routes ...gatewayrecovery.Route) *gatewayrecovery.Participant {
	t.Helper()
	if _, err := f.runs.CreateRun(t.Context(), runstore.CreateRunParams{RunID: id, OwnerID: "owner", WorkflowName: "test", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`), RuntimeConfig: runtimeconfig.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	if _, err := f.runs.TransitionRun(t.Context(), id, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	allowed, err := f.service.Admit(t.Context(), id, routes)
	if err != nil {
		t.Fatal(err)
	}
	if allowed {
		if _, err := f.runs.TransitionRun(t.Context(), id, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
			t.Fatal(err)
		}
	}
	return f.service.Planner(id, id, routes[0], contracts.DefaultGatewayFailureSignatures())
}
func (f recoveryFixture) due(t *testing.T) {
	t.Helper()
	if _, err := f.pool.Exec(t.Context(), `UPDATE gateway_recovery_routes SET next_probe_at=clock_timestamp()-interval '1 second'`); err != nil {
		t.Fatal(err)
	}
}
func recoveryUpdate(t *testing.T, p *gatewayrecovery.Participant, id, action string) gatewayrecovery.Decision {
	t.Helper()
	request := gatewayrecovery.Request{RequestID: id, Action: action}
	if action == "failed" {
		request.Code = "model_unavailable"
	}
	decision, err := p.Update(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	return decision
}

func TestGatewayRecoveryAcrossLanesPreservesQueueAndIndependentRoutes(t *testing.T) {
	f := newRecoveryFixture(t)
	first := f.run(t, "active-one", f.route)
	second := f.run(t, "active-two", f.route)
	recoveryUpdate(t, first, "outage", "failed")
	recoveryUpdate(t, second, "other-waiter", "acquire")
	f.run(t, "queued", f.route)
	queued, err := f.runs.GetRun(t.Context(), "queued")
	if err != nil || queued.State != runstore.RunPending || queued.StartedAt != nil {
		t.Fatalf("queued Run was admitted: %+v %v", queued, err)
	}
	independent := f.route
	independent.Model = "other-model"
	free := f.run(t, "independent", independent)
	if !recoveryUpdate(t, free, "healthy", "acquire").Allowed {
		t.Fatal("outage blocked independent model")
	}
	f.due(t)
	participants := []*gatewayrecovery.Participant{first, second}
	var wait sync.WaitGroup
	decisions := make([]gatewayrecovery.Decision, len(participants))
	errors := make([]error, len(participants))
	for i, p := range participants {
		wait.Add(1)
		go func() {
			defer wait.Done()
			decisions[i], errors[i] = p.Update(t.Context(), gatewayrecovery.Request{RequestID: fmt.Sprint("probe-", i), Action: "acquire"})
		}()
	}
	wait.Wait()
	winner := -1
	for i, d := range decisions {
		if errors[i] != nil {
			t.Fatal(errors[i])
		}
		if d.Allowed {
			if winner != -1 {
				t.Fatal("more than one probe admitted")
			}
			winner = i
		}
	}
	if winner < 0 {
		t.Fatal("no probe admitted")
	}
	recoveryUpdate(t, first, "late-old-request", "succeeded")
	if allowed, _ := f.service.Admit(t.Context(), "queued", []gatewayrecovery.Route{f.route}); allowed {
		t.Fatal("late response reopened gate")
	}
	recoveryUpdate(t, participants[winner], fmt.Sprint("probe-", winner), "succeeded")
	if allowed, err := f.service.Admit(t.Context(), "queued", []gatewayrecovery.Route{f.route}); err != nil || !allowed {
		t.Fatalf("recovered queue blocked: %v %v", allowed, err)
	}
}

func TestGatewayRecoveryAutomaticWindowManualRetryAndCancellation(t *testing.T) {
	f := newRecoveryFixture(t)
	participant := f.run(t, "waiting", f.route)
	recoveryUpdate(t, participant, "outage", "failed")
	if _, err := f.pool.Exec(t.Context(), `UPDATE gateway_recovery_routes SET automatic_until=clock_timestamp()-interval '1 second'`); err != nil {
		t.Fatal(err)
	}
	decision := recoveryUpdate(t, participant, "probe", "acquire")
	if decision.Allowed || !decision.RequiresRetry {
		t.Fatalf("exhausted window: %+v", decision)
	}
	status, err := f.service.Status(t.Context(), "waiting")
	if err != nil || status == nil || !status.RequiresRetry || status.NextRetryAt != nil {
		t.Fatalf("public status: %+v %v", status, err)
	}
	if err := f.service.Retry(t.Context(), "another-owner", "waiting"); err == nil {
		t.Fatal("foreign owner resumed route")
	}
	if err := f.service.Retry(t.Context(), "owner", "waiting"); err != nil {
		t.Fatal(err)
	}
	if !recoveryUpdate(t, participant, "manual-probe", "acquire").Allowed {
		t.Fatal("manual retry did not enable probe")
	}
	if _, err := f.runs.RequestRunCancellation(t.Context(), "waiting", runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}
	if _, err := participant.Update(t.Context(), gatewayrecovery.Request{RequestID: "after-cancel", Action: "acquire"}); !gatewayrecovery.IsUnavailable(err) {
		t.Fatalf("cancelled invocation retained authority: %v", err)
	}
}

func TestGatewayRecoveryMultiRouteAdmissionDoesNotLeakProbe(t *testing.T) {
	f := newRecoveryFixture(t)
	other := f.route
	other.Model = "other"
	first := f.run(t, "first", f.route)
	second := f.run(t, "second", other)
	recoveryUpdate(t, first, "failure-a", "failed")
	recoveryUpdate(t, second, "failure-b", "failed")
	// No live waiter on A; B still has an active waiter.
	if _, err := f.runs.TransitionRun(t.Context(), "first", runstore.RunWaiting, runstore.RunFailed, runstore.Reason{Code: "test_ended"}); err != nil {
		t.Fatal(err)
	}
	f.due(t)
	f.run(t, "queued", f.route, other)
	var probes int
	if err := f.pool.QueryRow(t.Context(), `SELECT count(*) FROM gateway_recovery_routes WHERE probe_run_id='queued'`).Scan(&probes); err != nil || probes != 0 {
		t.Fatalf("partial admission leaked probe: %d %v", probes, err)
	}
}

func TestGatewayRecoveryRequestRetryHintAndFailureReplay(t *testing.T) {
	f := newRecoveryFixture(t)
	participant := f.run(t, "waiting", f.route)
	request := gatewayrecovery.Request{RequestID: "outage", Action: "failed", Code: "gateway_rate_limited", RetryAfterSeconds: 90}
	if _, err := participant.Update(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	other := request
	other.RequestID = "interleaved"
	if _, err := participant.Update(context.Background(), other); err != nil {
		t.Fatal(err)
	}
	if _, err := participant.Update(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	var failures int
	var next time.Time
	if err := f.pool.QueryRow(t.Context(), `SELECT failure_count,next_probe_at FROM gateway_recovery_routes`).Scan(&failures, &next); err != nil {
		t.Fatal(err)
	}
	if failures != 2 || time.Until(next) < 89*time.Second {
		t.Fatalf("replay/hint: failures=%d delay=%s", failures, time.Until(next))
	}
}

func TestGatewayWaitingRunAcceptsStageResultAndDropsWaits(t *testing.T) {
	ctx := t.Context()
	pool := isolatedSchedulerPool(t, ctx)
	service, err := gatewayrecovery.New(pool, gatewayrecovery.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	route := gatewayrecovery.Route{OwnerID: "user-1", GatewayDigest: "gateway", Model: "model"}
	fixture := createFinalizingFixtureWith(t, ctx, pool, func() {
		if allowed, err := service.Admit(ctx, "run-1", []gatewayrecovery.Route{route}); err != nil || !allowed {
			t.Fatalf("admit route = %v, %v", allowed, err)
		}
		participant := service.Planner("run-1", "invocation-finalizing", route, contracts.DefaultGatewayFailureSignatures())
		recoveryUpdate(t, participant, "outage", "failed")
		run, err := runstore.NewPostgresStore(pool).GetRun(ctx, "run-1")
		if err != nil || run.State != runstore.RunWaiting {
			t.Fatalf("Run before result = (%+v, %v), want waiting", run, err)
		}
	})
	if err := fixture.persistence.CommitResultProgression(ctx, ResultProgression{
		RunID: "run-1", StageExecutionID: fixture.executionID, Result: fixture.result,
		WorkflowOutputs: fixture.workflow.Stages["copy"].WorkflowOutputs,
		OutputContracts: fixture.workflow.Outputs,
		Progression:     terminalSuccessProgression("run-1", fixture.executionID),
	}); err != nil {
		t.Fatalf("commit result for waiting Run: %v", err)
	}
	run, err := fixture.store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunSucceeded {
		t.Fatalf("Run after result = (%+v, %v)", run, err)
	}
	var waits int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM gateway_recovery_waits WHERE run_id='run-1'`).Scan(&waits); err != nil || waits != 0 {
		t.Fatalf("gateway waits after result = %d, %v", waits, err)
	}
}
