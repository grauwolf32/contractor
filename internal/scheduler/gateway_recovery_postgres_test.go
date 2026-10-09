package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
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
	service, err := gatewayrecovery.New(pool, gatewayrecovery.DefaultPolicy(), time.Minute)
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
	return f.service.Planner(id, id, routes[0], llmgateway.DefaultGatewayFailureSignatures())
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
	status, err := f.service.Status(t.Context(), "queued")
	if err != nil || status == nil || status.Code != "model_unavailable" {
		t.Fatalf("denied pending Run lost its current blocked route: %+v %v", status, err)
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

func TestGatewayRecoveryDropsPreviousStageRoutesOnProgression(t *testing.T) {
	ctx := t.Context()
	pool := isolatedSchedulerPool(t, ctx)
	service, err := gatewayrecovery.New(pool, gatewayrecovery.DefaultPolicy(), time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	oldRoute := gatewayrecovery.Route{OwnerID: "user-1", GatewayDigest: "gateway", Model: "model-a"}
	newRoute := oldRoute
	newRoute.Model = "model-b"
	fixture := createFinalizingFixtureWith(t, ctx, pool, func() {
		if allowed, err := service.Admit(ctx, "run-1", []gatewayrecovery.Route{oldRoute}); err != nil || !allowed {
			t.Fatalf("admit first Stage route = %v, %v", allowed, err)
		}
		participant := service.Planner("run-1", "invocation-finalizing", oldRoute, llmgateway.DefaultGatewayFailureSignatures())
		recoveryUpdate(t, participant, "old-outage", "failed")
	})
	var beforeAutomatic, beforeNext time.Time
	if err := pool.QueryRow(ctx, `
SELECT automatic_until,next_probe_at FROM gateway_recovery_routes WHERE route_key=$1`, oldRoute.Key()).Scan(
		&beforeAutomatic, &beforeNext,
	); err != nil {
		t.Fatal(err)
	}
	stage := fixture.workflow.Stages["copy"]
	stageJSON, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	runArtifacts, err := fixture.artifacts.Run("run-1")
	if err != nil {
		t.Fatal(err)
	}
	input, err := runArtifacts.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil {
		t.Fatal(err)
	}
	const nextID = "stage-next-route"
	nextStageName := "copy"
	previous := fixture.executionID
	progression := StageProgression{
		Decision: runstore.RecordStageTransitionDecisionParams{
			SourceExecutionID: fixture.executionID, RunID: "run-1", Action: runstore.StageTransitionNext,
			TargetStageName: &nextStageName, TargetExecutionID: stringPointer(nextID),
		},
		NextStage: &NextStageCreation{
			Params: runstore.CreateStageExecutionParams{
				StageExecutionID: nextID, RunID: "run-1", StageName: nextStageName,
				Attempt: 2, PreviousExecutionID: &previous,
				ExecutionConfigVariant: runstore.StageExecutionConfigBase,
				StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
				StageContextSchemaVersion: contracts.APIVersion,
				StageContext: runstore.StageContextSnapshot{
					Parameters: map[string]string{"objective": "copy exactly"},
					Artifacts: map[string]runstore.PinnedContextArtifact{
						"source": {Required: true, Artifact: &input.Ref},
					},
				},
			},
			ContextPins: []ContextPin{{Name: "source", Ref: input.Ref}},
		},
	}
	if err := fixture.persistence.CommitResultProgression(ctx, ResultProgression{
		RunID: "run-1", StageExecutionID: fixture.executionID, Result: fixture.result,
		WorkflowOutputs: stage.WorkflowOutputs, OutputContracts: fixture.workflow.Outputs,
		Progression: progression,
	}); err != nil {
		t.Fatal(err)
	}
	var remainingRoutes int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM gateway_run_routes WHERE run_id='run-1'`).Scan(
		&remainingRoutes,
	); err != nil || remainingRoutes != 0 {
		t.Fatalf("completed Stage retained %d route memberships: %v", remainingRoutes, err)
	}
	if allowed, err := service.Admit(ctx, "run-1", []gatewayrecovery.Route{newRoute}); err != nil || !allowed {
		t.Fatalf("admit next Stage route = %v, %v", allowed, err)
	}
	if status, err := service.Status(ctx, "run-1"); err != nil || status != nil {
		t.Fatalf("old blocked route leaked into next Stage status: %+v %v", status, err)
	}
	if err := service.Retry(ctx, "user-1", "run-1"); !gatewayrecovery.IsUnavailable(err) {
		t.Fatalf("retry reopened previous Stage route: %v", err)
	}
	var afterAutomatic, afterNext time.Time
	if err := pool.QueryRow(ctx, `
SELECT automatic_until,next_probe_at FROM gateway_recovery_routes WHERE route_key=$1`, oldRoute.Key()).Scan(
		&afterAutomatic, &afterNext,
	); err != nil {
		t.Fatal(err)
	}
	if !beforeAutomatic.Equal(afterAutomatic) || !beforeNext.Equal(afterNext) {
		t.Fatalf("old route changed on unrelated retry: automatic %s -> %s, next %s -> %s",
			beforeAutomatic, afterAutomatic, beforeNext, afterNext)
	}
	var keys []string
	rows, err := pool.Query(ctx, `SELECT route_key FROM gateway_run_routes WHERE run_id='run-1' ORDER BY route_key`)
	if err != nil {
		t.Fatal(err)
	}
	for rows.Next() {
		var key string
		if err := rows.Scan(&key); err != nil {
			t.Fatal(err)
		}
		keys = append(keys, key)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	rows.Close()
	if !reflect.DeepEqual(keys, []string{newRoute.Key()}) {
		t.Fatalf("next Stage route membership = %v", keys)
	}
}

func TestGatewayRecoveryAdmissionReplacesEarlierCandidateRoutes(t *testing.T) {
	f := newRecoveryFixture(t)
	participant := f.run(t, "blocking", f.route)
	recoveryUpdate(t, participant, "outage", "failed")
	f.run(t, "candidate", f.route)
	if status, err := f.service.Status(t.Context(), "candidate"); err != nil || status == nil {
		t.Fatalf("denied pending admission has no blocked route: %+v %v", status, err)
	}
	other := f.route
	other.Model = "other-model"
	if allowed, err := f.service.Admit(t.Context(), "candidate", []gatewayrecovery.Route{other}); err != nil || !allowed {
		t.Fatalf("replacement candidate was not admitted: %t %v", allowed, err)
	}
	if status, err := f.service.Status(t.Context(), "candidate"); err != nil || status != nil {
		t.Fatalf("earlier candidate's blocked route leaked into status: %+v %v", status, err)
	}
	var count int
	if err := f.pool.QueryRow(t.Context(), `
SELECT count(*) FROM gateway_run_routes WHERE run_id='candidate' AND route_key=$1`, f.route.Key()).Scan(
		&count,
	); err != nil || count != 0 {
		t.Fatalf("earlier candidate still belongs to Run: %d %v", count, err)
	}
}

func TestGatewayRecoveryAbandonedProbePreservesBlockedRoute(t *testing.T) {
	f := newRecoveryFixture(t)
	first := f.run(t, "active-one", f.route)
	second := f.run(t, "active-two", f.route)
	recoveryUpdate(t, first, "outage", "failed")
	f.due(t)
	if !recoveryUpdate(t, first, "abandoned-probe", "acquire").Allowed {
		t.Fatal("first participant did not acquire the due probe")
	}
	if recoveryUpdate(t, second, "waiting-request", "acquire").Allowed {
		t.Fatal("second participant acquired the same probe")
	}
	type routeSnapshot struct {
		blocked                             bool
		failures                            int64
		blockedAt, next, automatic, probeAt time.Time
		probeID, probeRunID                 *string
	}
	read := func() routeSnapshot {
		t.Helper()
		var state routeSnapshot
		err := f.pool.QueryRow(t.Context(), `
SELECT blocked,failure_count,blocked_at,next_probe_at,automatic_until,
       probe_until,probe_id,probe_run_id FROM gateway_recovery_routes`).Scan(
			&state.blocked, &state.failures, &state.blockedAt, &state.next,
			&state.automatic, &state.probeAt, &state.probeID, &state.probeRunID)
		if err != nil {
			t.Fatal(err)
		}
		return state
	}
	before := read()
	recoveryUpdate(t, second, "stale-request", "released")
	if got := read(); !reflect.DeepEqual(got, before) {
		t.Fatalf("foreign release changed route: before=%+v after=%+v", before, got)
	}
	recoveryUpdate(t, first, "abandoned-probe", "released")
	var blocked bool
	var failures int64
	var blockedAt, next, automatic time.Time
	var probeID, probeRunID *string
	var probeUntil *time.Time
	if err := f.pool.QueryRow(t.Context(), `
SELECT blocked,failure_count,blocked_at,next_probe_at,automatic_until,
       probe_until,probe_id,probe_run_id FROM gateway_recovery_routes`).Scan(
		&blocked, &failures, &blockedAt, &next, &automatic,
		&probeUntil, &probeID, &probeRunID); err != nil {
		t.Fatal(err)
	}
	if !blocked || failures != before.failures || !blockedAt.Equal(before.blockedAt) ||
		!next.Equal(before.next) || !automatic.Equal(before.automatic) ||
		probeID != nil || probeRunID != nil || probeUntil != nil {
		t.Fatalf("abandoned probe reopened or reset route: blocked=%v failures=%d probe=%v", blocked, failures, probeID)
	}
	var waits int
	if err := f.pool.QueryRow(t.Context(), `SELECT count(*) FROM gateway_recovery_waits`).Scan(&waits); err != nil || waits != 1 {
		t.Fatalf("release removed another participant's wait: count=%d err=%v", waits, err)
	}
	if !recoveryUpdate(t, second, "replacement-probe", "acquire").Allowed {
		t.Fatal("waiting participant could not acquire released probe")
	}
	recoveryUpdate(t, first, "abandoned-probe", "released")
	if got := read(); got.probeID == nil || *got.probeID != "replacement-probe" {
		t.Fatalf("stale release cleared replacement probe: %+v", got)
	}
	recoveryUpdate(t, second, "replacement-probe", "finished")
	if err := f.pool.QueryRow(t.Context(), `SELECT blocked,failure_count,probe_id FROM gateway_recovery_routes`).Scan(
		&blocked, &failures, &probeID); err != nil {
		t.Fatal(err)
	}
	if blocked || failures != 0 || probeID != nil {
		t.Fatalf("observed permanent response did not reopen route: blocked=%v failures=%d probe=%v", blocked, failures, probeID)
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
	service, err := gatewayrecovery.New(pool, gatewayrecovery.DefaultPolicy(), time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	route := gatewayrecovery.Route{OwnerID: "user-1", GatewayDigest: "gateway", Model: "model"}
	fixture := createFinalizingFixtureWith(t, ctx, pool, func() {
		if allowed, err := service.Admit(ctx, "run-1", []gatewayrecovery.Route{route}); err != nil || !allowed {
			t.Fatalf("admit route = %v, %v", allowed, err)
		}
		participant := service.Planner("run-1", "invocation-finalizing", route, llmgateway.DefaultGatewayFailureSignatures())
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
