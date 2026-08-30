package runstore

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresIntegrationStageLifecycleAndSessions(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)

	run := createTestRun(t, ctx, store, "run-lifecycle")
	if run.State != RunInitializing || run.WorkflowSnapshot == nil {
		t.Fatalf("created Run = %+v", run)
	}
	run, err := store.TransitionRun(ctx, run.RunID, RunInitializing, RunRunning, Reason{Code: "initialized"})
	if err != nil {
		t.Fatalf("start Run: %v", err)
	}
	if run.State != RunRunning || run.StartedAt == nil {
		t.Fatalf("started Run = %+v", run)
	}

	revision := "revision-1"
	execution, err := store.CreateStageExecution(ctx, CreateStageExecutionParams{
		StageExecutionID: "stage-lifecycle", RunID: run.RunID, StageName: "copy", Attempt: 1,
		StageSpecSchemaVersion:    "contractor/v1alpha1",
		StageSpecSnapshot:         json.RawMessage(`{"objective":"copy"}`),
		StageContextSchemaVersion: "contractor/v1alpha1",
		StageContext: StageContextSnapshot{
			Parameters: map[string]string{"mode": "strict"},
			Artifacts: map[string]PinnedContextArtifact{
				"source": {
					Required: true,
					Artifact: &contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &revision},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("create StageExecution: %v", err)
	}
	if execution.State != StagePreparing || execution.PlannerSessionID != nil {
		t.Fatalf("preparing StageExecution = %+v", execution)
	}

	allocation := StageAllocation{
		AllocationID: "allocation-1", StageExecutionID: execution.StageExecutionID,
		LogicalAgentName: "builder", Namespace: "builder",
		AgentTemplateRef: contracts.AgentTemplateRef{
			TemplateID: "artifact_builder", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
		},
		WorkerRuntimeRef:       contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"},
		RuntimeAgentInstanceID: "runtime-agent-1",
	}
	err = store.RecordStageAllocation(ctx, allocation)
	if err != nil {
		t.Fatalf("record allocation: %v", err)
	}
	if err := store.RecordStageAllocation(ctx, allocation); err != nil {
		t.Fatalf("idempotent allocation record: %v", err)
	}
	different := allocation
	different.RuntimeAgentInstanceID = "another-runtime-agent"
	if err := store.RecordStageAllocation(ctx, different); !errors.Is(err, ErrConflict) {
		t.Fatalf("mismatched allocation record error = %v, want conflict", err)
	}
	allocations, err := store.ListStageAllocations(ctx, execution.StageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].AllocationID != "allocation-1" {
		t.Fatalf("allocations = (%+v, %v)", allocations, err)
	}
	reportFinished := time.Now().UTC()
	modelCalls := int64(3)
	reportParams := RecordStageExecutionReportParams{
		StageExecutionID: execution.StageExecutionID, AllocationID: allocation.AllocationID,
		LogicalAgentName: allocation.LogicalAgentName, ReportSchemaVersion: contracts.APIVersion,
		Report: contracts.AllocationFinalReport{
			ReportID: "allocation-report-1", AllocationID: allocation.AllocationID,
			StartedAt: reportFinished.Add(-time.Second), FinishedAt: reportFinished,
			Worker: contracts.ExecutionReport{
				ReportID: "worker-report-1", Complete: true,
				Metrics: contracts.ExecutionMetrics{
					ModelCalls: &modelCalls, Tools: map[string]contracts.ToolMetrics{},
				},
				ToolCalls: []contracts.ToolCallRecord{}, Errors: []contracts.ExecutionError{},
			},
			Runtime: contracts.RuntimeReport{Complete: true},
		},
	}
	if err := store.RecordStageExecutionReport(ctx, reportParams); err != nil {
		t.Fatalf("record execution report: %v", err)
	}
	if err := store.RecordStageExecutionReport(ctx, reportParams); err != nil {
		t.Fatalf("idempotent execution report: %v", err)
	}
	reports, err := store.ListStageExecutionReports(ctx, execution.StageExecutionID)
	if err != nil || len(reports) != 1 || reports[0].Report.Worker.Metrics.ModelCalls == nil ||
		*reports[0].Report.Worker.Metrics.ModelCalls != 3 {
		t.Fatalf("execution reports = (%+v, %v)", reports, err)
	}
	differentReport := reportParams
	differentCalls := int64(4)
	differentReport.Report.Worker.Metrics.ModelCalls = &differentCalls
	if err := store.RecordStageExecutionReport(ctx, differentReport); !errors.Is(err, ErrConflict) {
		t.Fatalf("mismatched execution report error = %v, want conflict", err)
	}
	if _, err := pool.Exec(ctx, `
UPDATE allocation_execution_reports
SET report = jsonb_set(report, '{worker,complete}', 'false')
WHERE allocation_id = 'allocation-1'`); persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("execution report rewrite SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}

	startedRunEvent := RunEventAppend{
		EventID: "event-started", EventSchemaVersion: contracts.APIVersion,
		Kind: RunEventPlannerStarted,
		Data: mustRunEventJSON(t, plannerRunEventData{
			StageExecutionID: execution.StageExecutionID,
			SessionID:        "session-1", InvocationID: "invocation-1",
		}),
	}
	err = store.StartPlanner(ctx, StartPlannerParams{
		StageExecutionID: execution.StageExecutionID,
		SessionID:        "session-1", InvocationID: "invocation-1",
		StateSchemaVersion: contracts.APIVersion, InitialState: json.RawMessage(`{"step":0}`),
		EventID: "event-started", EventSchemaVersion: contracts.APIVersion,
		Event: json.RawMessage(`{"kind":"planner_started"}`), RunEvent: startedRunEvent,
		Reason: Reason{Code: "planner_started"},
	})
	if err != nil {
		t.Fatalf("start Planner: %v", err)
	}
	execution, err = store.GetStageExecution(ctx, execution.StageExecutionID)
	if err != nil || execution.State != StageRunning || execution.PlannerSessionID == nil || *execution.PlannerSessionID != "session-1" {
		t.Fatalf("running StageExecution = (%+v, %v)", execution, err)
	}

	requestRunEvent := RunEventAppend{
		EventID: "event-1", EventSchemaVersion: contracts.APIVersion,
		Kind: RunEventPlannerRequestRecorded,
		Data: mustRunEventJSON(t, plannerRunEventData{
			StageExecutionID: execution.StageExecutionID,
			SessionID:        "session-1", InvocationID: "invocation-1",
		}),
	}
	err = store.AppendPlannerEvent(ctx, AppendPlannerEventParams{
		EventID: "event-1", SessionID: "session-1",
		StageExecutionID: execution.StageExecutionID, InvocationID: "invocation-1",
		SequenceNumber:     2,
		EventSchemaVersion: contracts.APIVersion, Event: json.RawMessage(`{"kind":"worker_request"}`),
		NewStateSchemaVersion: contracts.APIVersion, NewState: json.RawMessage(`{"step":1}`),
		RunEvent: requestRunEvent,
	})
	if err != nil {
		t.Fatalf("append Planner event: %v", err)
	}
	session, err := store.GetPlannerSession(ctx, "session-1")
	if err != nil || session.NextEventSequence != 3 ||
		string(session.State) != `{"step": 1}` && string(session.State) != `{"step":1}` {
		t.Fatalf("Planner session = (%+v, %v)", session, err)
	}
	events, err := store.ListPlannerEvents(ctx, "session-1", 0)
	if err != nil || len(events) != 2 || events[0].EventID != "event-started" ||
		events[1].EventID != "event-1" || events[0].RunEventSequence == nil ||
		*events[0].RunEventSequence != 5 || events[1].RunEventSequence == nil ||
		*events[1].RunEventSequence != 6 {
		t.Fatalf("Planner events = (%+v, %v)", events, err)
	}
	gap := requestRunEvent
	gap.EventID = "event-gap"
	err = store.AppendPlannerEvent(ctx, AppendPlannerEventParams{
		EventID: "event-gap", SessionID: "session-1",
		StageExecutionID: execution.StageExecutionID, InvocationID: "invocation-1",
		SequenceNumber: 4, EventSchemaVersion: contracts.APIVersion,
		Event:                 json.RawMessage(`{"kind":"worker_request"}`),
		NewStateSchemaVersion: contracts.APIVersion, NewState: json.RawMessage(`{"step":2}`),
		RunEvent: gap,
	})
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("Planner sequence gap error = %v, want conflict", err)
	}
	cursor, err := store.GetRunEventCursor(ctx, run.RunID)
	if err != nil || cursor.Sequence != 6 || cursor.Generation == "" {
		t.Fatalf("Run event cursor = (%+v, %v)", cursor, err)
	}
	runEvents, err := store.ListRunEvents(ctx, run.RunID, 0, 10)
	if err != nil || len(runEvents) != 6 || runEvents[0].Kind != RunEventLifecycleChanged ||
		runEvents[1].Kind != RunEventLifecycleChanged || runEvents[2].Kind != RunEventLifecycleChanged ||
		runEvents[3].Kind != RunEventLifecycleChanged || runEvents[4].Kind != RunEventPlannerStarted ||
		runEvents[5].Kind != RunEventPlannerRequestRecorded {
		t.Fatalf("Run events = (%+v, %v)", runEvents, err)
	}
	wantLifecycle := []lifecycleRunEventData{
		{RunID: run.RunID, Resource: "run", State: string(RunInitializing)},
		{RunID: run.RunID, Resource: "run", State: string(RunRunning)},
		{RunID: run.RunID, Resource: "stageExecution", StageExecutionID: execution.StageExecutionID, State: string(StagePreparing)},
		{RunID: run.RunID, Resource: "stageExecution", StageExecutionID: execution.StageExecutionID, State: string(StageRunning)},
	}
	for index, want := range wantLifecycle {
		got, decodeErr := decodeLifecycleRunEventData(runEvents[index].Data)
		if decodeErr != nil || got != want || runEvents[index].SequenceNumber != int64(index+1) {
			t.Fatalf("lifecycle event %d = (%+v, %v), want %+v", index, got, decodeErr, want)
		}
	}

	resultRevision := "revision-result-1"
	candidate := contracts.StageContentResult{
		APIVersion: contracts.APIVersion,
		Outcome:    contracts.StageSucceeded,
		Summary:    "copied",
		Artifacts: map[string]contracts.ArtifactRef{
			"copied": {Namespace: "builder", Name: "result", Revision: &resultRevision},
		},
	}
	err = store.EnterFinalizing(ctx, EnterFinalizingParams{
		StageExecutionID:    execution.StageExecutionID,
		ResultSchemaVersion: "contractor/v1alpha1", Candidate: candidate,
		FinalizationID: "finalization-1", Deadline: time.Now().Add(time.Minute),
		Reason: Reason{Code: "planner_completed"},
	})
	if err != nil {
		t.Fatalf("enter finalizing: %v", err)
	}
	err = store.EnterAborting(ctx, EnterAbortingParams{
		StageExecutionID: execution.StageExecutionID, ExpectedState: StageRunning,
		TerminationSchemaVersion: "contractor/v1alpha1",
		Termination: StageTermination{
			Outcome: TerminationInterrupted, Code: "late_abort", Message: "late abort",
			Retryable: true, Phase: TerminationRunning, OccurredAt: time.Now(),
		},
		AbortID: "late-abort", Deadline: time.Now().Add(time.Minute), Reason: Reason{Code: "late_abort"},
	})
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("late abort error = %v, want conflict", err)
	}
	if err := store.CompleteStageResult(ctx, execution.StageExecutionID, "contractor/v1alpha1", candidate); err != nil {
		t.Fatalf("complete StageResult: %v", err)
	}
	execution, err = store.GetStageExecution(ctx, execution.StageExecutionID)
	if err != nil || execution.State != StageSucceeded || execution.CandidateResult == nil || execution.AcceptedResult == nil {
		t.Fatalf("terminal StageExecution = (%+v, %v)", execution, err)
	}
	if got := *execution.AcceptedResult.Artifacts["copied"].Revision; got != resultRevision {
		t.Fatalf("accepted artifact revision = %q, want %q", got, resultRevision)
	}
	if err := store.CompleteStageResult(ctx, execution.StageExecutionID, "contractor/v1alpha1", candidate); !errors.Is(err, ErrConflict) {
		t.Fatalf("second completion error = %v, want conflict", err)
	}

	_, err = store.CreateStageExecution(ctx, CreateStageExecutionParams{
		StageExecutionID: "stage-aborted", RunID: run.RunID, StageName: "precheck", Attempt: 1,
		StageSpecSchemaVersion: "contractor/v1alpha1", StageSpecSnapshot: json.RawMessage(`{"objective":"check"}`),
		StageContextSchemaVersion: "contractor/v1alpha1", StageContext: StageContextSnapshot{},
	})
	if err != nil {
		t.Fatalf("create aborting StageExecution: %v", err)
	}
	termination := StageTermination{
		Outcome: TerminationInterrupted, Code: "context_artifact_missing", Message: "source is missing",
		Retryable: false, Phase: TerminationPreparing, OccurredAt: time.Now(),
	}
	err = store.EnterAborting(ctx, EnterAbortingParams{
		StageExecutionID: "stage-aborted", ExpectedState: StagePreparing,
		TerminationSchemaVersion: "contractor/v1alpha1", Termination: termination,
		AbortID: "abort-1", Deadline: time.Now().Add(time.Minute), Reason: Reason{Code: termination.Code},
	})
	if err != nil {
		t.Fatalf("enter aborting during preparation: %v", err)
	}
	if err := store.CompleteStageTermination(ctx, "stage-aborted"); err != nil {
		t.Fatalf("complete StageTermination: %v", err)
	}
	aborted, err := store.GetStageExecution(ctx, "stage-aborted")
	if err != nil || aborted.State != StageInterrupted || aborted.Termination == nil || aborted.PlannerSessionID != nil {
		t.Fatalf("interrupted StageExecution = (%+v, %v)", aborted, err)
	}

	executions, err := store.ListStageExecutions(ctx, run.RunID)
	if err != nil || len(executions) != 2 {
		t.Fatalf("StageExecution list = (%+v, %v)", executions, err)
	}

	_, err = pool.Exec(ctx, `
UPDATE stage_executions
SET candidate_stage_result = jsonb_set(candidate_stage_result, '{summary}', '"rewritten"')
WHERE stage_execution_id = 'stage-lifecycle'`)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("candidate rewrite SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
}

func TestPostgresIntegrationClaimsConflictsAndExplicitTransactions(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	createTestRun(t, ctx, store, "run-race")

	var transitions int32
	var conflicts int32
	var wait sync.WaitGroup
	for range 2 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, err := store.TransitionRun(ctx, "run-race", RunInitializing, RunRunning, Reason{Code: "ready"})
			switch {
			case err == nil:
				atomic.AddInt32(&transitions, 1)
			case errors.Is(err, ErrConflict):
				atomic.AddInt32(&conflicts, 1)
			default:
				t.Errorf("transition race: %v", err)
			}
		}()
	}
	wait.Wait()
	if transitions != 1 || conflicts != 1 {
		t.Fatalf("transition race successes=%d conflicts=%d", transitions, conflicts)
	}

	type claimResult struct {
		run WorkflowRun
		err error
	}
	claimResults := make(chan claimResult, 2)
	for _, claimID := range []string{"claim-a", "claim-b"} {
		wait.Add(1)
		go func(claimID string) {
			defer wait.Done()
			run, err := store.ClaimRunnableRun(ctx, claimID, time.Minute)
			claimResults <- claimResult{run: run, err: err}
		}(claimID)
	}
	wait.Wait()
	close(claimResults)
	var claimed WorkflowRun
	var claims, noWork int
	for result := range claimResults {
		if result.err == nil {
			claims++
			claimed = result.run
		} else if errors.Is(result.err, ErrNoWork) {
			noWork++
		} else {
			t.Fatalf("claim race: %v", result.err)
		}
	}
	if claims != 1 || noWork != 1 || claimed.SchedulerClaim == nil {
		t.Fatalf("claim race claims=%d noWork=%d claimed=%+v", claims, noWork, claimed)
	}
	if err := store.RenewRunClaim(ctx, claimed.RunID, claimed.SchedulerClaim.ClaimID, 2*time.Minute); err != nil {
		t.Fatalf("renew claim: %v", err)
	}
	if err := store.ReleaseRunClaim(ctx, claimed.RunID, "wrong-claim"); !errors.Is(err, ErrConflict) {
		t.Fatalf("wrong release error = %v", err)
	}
	if err := store.ReleaseRunClaim(ctx, claimed.RunID, claimed.SchedulerClaim.ClaimID); err != nil {
		t.Fatalf("release claim: %v", err)
	}

	rollback := errors.New("force rollback")
	err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		txStore := NewPostgresStore(tx)
		if _, err := txStore.CreateRun(ctx, testRunParams("run-rolled-back")); err != nil {
			return err
		}
		return rollback
	})
	if !errors.Is(err, rollback) {
		t.Fatalf("transaction error = %v, want rollback sentinel", err)
	}
	if _, err := store.GetRun(ctx, "run-rolled-back"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("rolled-back Run lookup error = %v", err)
	}
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := NewPostgresStore(tx).CreateRun(ctx, testRunParams("run-committed"))
		return err
	})
	if err != nil {
		t.Fatalf("commit explicit transaction: %v", err)
	}
	if _, err := store.GetRun(ctx, "run-committed"); err != nil {
		t.Fatalf("get committed Run: %v", err)
	}
}

func TestWorkflowRunEventNotificationIsVisibleOnlyAfterCommit(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	listener, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Release()
	if _, err := listener.Exec(ctx, "LISTEN "+runEventNotificationChannel); err != nil {
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		t.Fatal(err)
	}
	runID := "run-notify-" + hex.EncodeToString(random)
	notified := make(chan struct{})
	waitErr := make(chan error, 1)
	go func() {
		for {
			notification, err := listener.Conn().WaitForNotification(ctx)
			if err != nil {
				waitErr <- err
				return
			}
			if notification.Channel == runEventNotificationChannel && notification.Payload == runID {
				close(notified)
				return
			}
		}
	}()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	_, err = NewPostgresStore(tx).CreateRun(ctx, CreateRunParams{
		RunID: runID, OwnerID: "user", WorkflowName: "workflow", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot:      json.RawMessage(`{"name":"workflow"}`),
		Parameters:            map[string]string{},
	})
	if err != nil {
		_ = tx.Rollback(ctx)
		t.Fatal(err)
	}
	select {
	case <-notified:
		_ = tx.Rollback(ctx)
		t.Fatal("WorkflowRun event notification escaped an uncommitted transaction")
	case err := <-waitErr:
		_ = tx.Rollback(ctx)
		t.Fatalf("wait before commit: %v", err)
	case <-time.After(100 * time.Millisecond):
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case <-notified:
	case err := <-waitErr:
		t.Fatalf("wait after commit: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("committed WorkflowRun event did not publish a wake-up notification")
	}
	events, err := NewPostgresStore(pool).ListRunEvents(ctx, runID, 0, 10)
	if err != nil || len(events) != 1 || events[0].Kind != RunEventLifecycleChanged ||
		events[0].SequenceNumber != 1 || !strings.Contains(string(events[0].Data), `"state": "initializing"`) {
		t.Fatalf("committed lifecycle events = (%+v, %v)", events, err)
	}
}

func TestPostgresIntegrationRunCancellationIsDurableAndIdempotent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)

	requestedBy := "user-1"
	reason := "stop this work"
	requestedAt := time.Date(2026, 8, 29, 12, 0, 0, 123, time.FixedZone("request-zone", 3*60*60))
	first, err := store.RequestRunCancellation(ctx, createTestRun(t, ctx, store, "run-cancel").RunID, WorkflowRunCancellation{
		Code: CancellationUserRequested, RequestedAt: requestedAt,
		RequestedBy: &requestedBy, Reason: &reason,
	})
	if err != nil {
		t.Fatalf("request cancellation: %v", err)
	}
	if first.State != RunCancelling || first.Cancellation == nil ||
		first.Cancellation.RequestedAt.Location() != time.UTC ||
		!first.Cancellation.RequestedAt.Equal(requestedAt) || first.Cancellation.Reason == nil ||
		*first.Cancellation.Reason != reason || first.CancellationSchemaVersion == nil {
		t.Fatalf("cancelling Run = %+v", first)
	}

	differentReason := "must not replace the winner"
	repeated, err := store.RequestRunCancellation(ctx, first.RunID, WorkflowRunCancellation{
		Code: CancellationUserRequested, RequestedAt: requestedAt.Add(time.Hour), Reason: &differentReason,
	})
	if err != nil {
		t.Fatalf("repeat cancellation: %v", err)
	}
	if repeated.Cancellation == nil || repeated.Cancellation.Reason == nil ||
		*repeated.Cancellation.Reason != reason || !repeated.Cancellation.RequestedAt.Equal(requestedAt) {
		t.Fatalf("repeated cancellation replaced the first payload: %+v", repeated.Cancellation)
	}

	claimed, err := store.ClaimRunnableRun(ctx, "cancel-claim", time.Minute)
	if err != nil || claimed.RunID != first.RunID || claimed.State != RunCancelling {
		t.Fatalf("claim cancelling Run = (%+v, %v)", claimed, err)
	}
	if err := store.RenewRunClaim(ctx, first.RunID, "cancel-claim", time.Minute); err != nil {
		t.Fatalf("renew cancelling Run claim: %v", err)
	}
	cancelled, err := store.TransitionRun(
		ctx, first.RunID, RunCancelling, RunCancelled, Reason{Code: CancellationUserRequested},
	)
	if err != nil || cancelled.State != RunCancelled || cancelled.FinishedAt == nil || cancelled.Cancellation == nil {
		t.Fatalf("finish cancelled Run = (%+v, %v)", cancelled, err)
	}

	_, err = pool.Exec(ctx, `
UPDATE workflow_runs
SET run_cancellation = jsonb_set(run_cancellation, '{reason}', '"rewritten"')
WHERE run_id = $1`, first.RunID)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("cancellation rewrite SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}

	succeeded := createTestRun(t, ctx, store, "run-succeeded-before-cancel")
	succeeded, err = store.TransitionRun(ctx, succeeded.RunID, RunInitializing, RunRunning, Reason{Code: "ready"})
	if err == nil {
		succeeded, err = store.TransitionRun(ctx, succeeded.RunID, RunRunning, RunSucceeded, Reason{Code: "completed"})
	}
	if err != nil {
		t.Fatalf("finish success before cancellation: %v", err)
	}
	afterCancel, err := store.RequestRunCancellation(ctx, succeeded.RunID, WorkflowRunCancellation{
		Code: CancellationUserRequested, RequestedAt: time.Now(),
	})
	if err != nil || afterCancel.State != RunSucceeded || afterCancel.Cancellation != nil {
		t.Fatalf("cancel terminal success = (%+v, %v)", afterCancel, err)
	}
}

func createTestRun(t *testing.T, ctx context.Context, store *PostgresStore, runID string) WorkflowRun {
	t.Helper()
	run, err := store.CreateRun(ctx, testRunParams(runID))
	if err != nil {
		t.Fatalf("create test Run %q: %v", runID, err)
	}
	return run
}

func testRunParams(runID string) CreateRunParams {
	return CreateRunParams{
		RunID: runID, OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: "contractor/v1alpha1",
		WorkflowSnapshot:      json.RawMessage(`{"ref":{"name":"artifact-copy","version":"1"}}`),
		Parameters:            map[string]string{"mode": "strict"},
	}
}

func isolatedRunStorePool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatalf("parse test database URL: %v", err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatalf("open test admin pool: %v", err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatalf("ping test database: %v", err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatalf("create isolated schema: %v", err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatalf("open isolated pool: %v", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		t.Fatalf("ping isolated pool: %v", err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		t.Fatalf("apply migrations: %v", err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop isolated schema: %v", err)
		}
		admin.Close()
	})
	return pool
}
