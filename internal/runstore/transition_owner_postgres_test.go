package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The Run store is the only writer of Run and Stage state and the only
// package that takes their row locks (PER-R23). These tests hold its
// transaction-bound commit operations to their compare-and-set rules on the
// real store; the Scheduler, recovery, Planner Memory and placement tests
// exercise them through their callers.

// inRunStoreTx runs fn with a store bound to a fresh transaction on pool.
func inRunStoreTx(ctx context.Context, pool *pgxpool.Pool, fn func(*PostgresTxStore) error) error {
	return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		return fn(NewPostgresTxStore(tx))
	})
}

// createStageInState creates StageExecution stageID of running Run runID and
// moves it into state through the store's own operations.
func createStageInState(
	t *testing.T, ctx context.Context, store *PostgresStore, runID, stageID string, state StageExecutionState,
) stageresult.StageContentResult {
	t.Helper()
	if _, err := store.CreateStageExecution(ctx, testActiveStageParams(runID, stageID, stageID)); err != nil {
		t.Fatal(err)
	}
	revision := "revision-" + stageID
	candidate := stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: "copied",
		Artifacts: map[string]contracts.ArtifactRef{
			"copied": {Namespace: "builder", Name: "result", Revision: &revision},
		},
	}
	if state == StagePreparing {
		return candidate
	}
	if state == StageAborting {
		if err := store.EnterAborting(ctx, EnterAbortingParams{
			StageExecutionID: stageID, ExpectedState: StagePreparing,
			TerminationSchemaVersion: contracts.APIVersion,
			Termination: StageTermination{
				Outcome: TerminationCancelled, Code: "stage_cancelled", Message: "cancelled",
				Phase: TerminationPreparing, OccurredAt: time.Now(),
			},
			AbortID: "abort-" + stageID, Deadline: time.Now().Add(time.Minute), Reason: Reason{Code: "stage_cancelled"},
		}); err != nil {
			t.Fatal(err)
		}
		return candidate
	}
	if err := store.StartPlanner(ctx, StartPlannerParams{
		StageExecutionID: stageID, SessionID: "session-" + stageID, InvocationID: "invocation-" + stageID,
		StateSchemaVersion: contracts.APIVersion, InitialState: json.RawMessage(`{}`),
		Reason: Reason{Code: "planner_started"},
		RunEvent: RunEventAppend{
			EventID: "event-" + stageID, EventSchemaVersion: contracts.APIVersion, Kind: RunEventPlannerStarted,
			Data: mustRunEventJSON(t, plannerRunEventData{
				StageExecutionID: stageID, SessionID: "session-" + stageID, InvocationID: "invocation-" + stageID,
			}),
		},
	}); err != nil {
		t.Fatal(err)
	}
	if state == StageRunning {
		return candidate
	}
	if err := store.EnterFinalizing(ctx, EnterFinalizingParams{
		StageExecutionID: stageID, ResultSchemaVersion: contracts.APIVersion, Candidate: candidate,
		FinalizationID: "finalization-" + stageID, Deadline: time.Now().Add(time.Minute),
		Reason: Reason{Code: "planner_completed"},
	}); err != nil {
		t.Fatal(err)
	}
	if state == StageFinalizing {
		return candidate
	}
	if err := store.CompleteStageResult(ctx, stageID, contracts.APIVersion, candidate); err != nil {
		t.Fatal(err)
	}
	return candidate
}

// TestPostgresRunAndStageLocksCheckTheirStates holds LockRunState and
// LockStageState to their checks and shows that each lock lasts until the
// transaction that took it ends.
func TestPostgresRunAndStageLocksCheckTheirStates(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	createTestRun(t, ctx, store, "run-locked")
	if _, err := admitTestRun(ctx, store, "run-locked"); err != nil {
		t.Fatal(err)
	}
	createTestRun(t, ctx, store, "run-other")
	createStageInState(t, ctx, store, "run-locked", "stage-locked", StagePreparing)

	err := inRunStoreTx(ctx, pool, func(locks *PostgresTxStore) error {
		if state, err := locks.LockRunState(ctx, "run-locked", RunPending, RunRunning); err != nil || state != RunRunning {
			t.Errorf("lock in an expected state = (%s, %v), want running", state, err)
		}
		if state, err := locks.LockRunState(ctx, "run-other"); err != nil || state != RunInitializing {
			t.Errorf("lock without expected states = (%s, %v), want initializing", state, err)
		}
		var conflict *StateConflictError
		if _, err := locks.LockRunState(ctx, "run-locked", RunWaiting, RunCancelling); !errors.As(err, &conflict) ||
			!errors.Is(err, ErrConflict) || conflict.Resource != "WorkflowRun" {
			t.Errorf("lock in an unexpected state = %v, want a WorkflowRun state conflict", err)
		}
		if _, err := locks.LockRunState(ctx, "run-missing"); !errors.Is(err, ErrNotFound) {
			t.Errorf("lock of a missing Run = %v, want not found", err)
		}
		if _, err := locks.LockRunState(ctx, " "); !errors.Is(err, ErrInvalid) {
			t.Errorf("lock without a Run ID = %v, want invalid", err)
		}

		if state, err := locks.LockStageState(ctx, "run-locked", "stage-locked", StagePreparing); err != nil ||
			state != StagePreparing {
			t.Errorf("Stage lock in an expected state = (%s, %v), want preparing", state, err)
		}
		if state, err := locks.LockStageState(ctx, "run-locked", "stage-locked"); err != nil || state != StagePreparing {
			t.Errorf("Stage lock without expected states = (%s, %v), want preparing", state, err)
		}
		if _, err := locks.LockStageState(ctx, "run-locked", "stage-locked", StageRunning); !errors.As(err, &conflict) ||
			conflict.Resource != "StageExecution" {
			t.Errorf("Stage lock in an unexpected state = %v, want a StageExecution state conflict", err)
		}
		if _, err := locks.LockStageState(ctx, "run-other", "stage-locked"); !errors.As(err, &conflict) ||
			conflict.Resource != "StageExecution Run" || conflict.Expected != "run-other" {
			t.Errorf("Stage lock under another Run = %v, want a StageExecution Run conflict", err)
		}
		if _, err := locks.LockStageState(ctx, "run-locked", "stage-missing"); !errors.Is(err, ErrNotFound) {
			t.Errorf("lock of a missing Stage = %v, want not found", err)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// Both locks are row locks held until the transaction ends: another
	// transaction cannot take them meanwhile, and can afterwards.
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	locks := NewPostgresTxStore(tx)
	if _, err := locks.LockRunState(ctx, "run-locked"); err != nil {
		t.Fatal(err)
	}
	if _, err := locks.LockStageState(ctx, "run-locked", "stage-locked"); err != nil {
		t.Fatal(err)
	}
	contended := map[string]string{
		"Run":   `SELECT 1 FROM workflow_runs WHERE run_id = 'run-locked' FOR UPDATE NOWAIT`,
		"Stage": `SELECT 1 FROM stage_executions WHERE stage_execution_id = 'stage-locked' FOR UPDATE NOWAIT`,
	}
	for name, query := range contended {
		if _, err := pool.Exec(ctx, query); persistencepostgres.SQLState(err) != persistencepostgres.SQLStateLockNotAvailable {
			t.Errorf("%s row lock taken while held: %v", name, err)
		}
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	for name, query := range contended {
		if _, err := pool.Exec(ctx, query); err != nil {
			t.Errorf("%s row lock after the transaction ended: %v", name, err)
		}
	}
}

// TestPostgresWaitingOperationsMoveOnlyAdmittedRuns holds the waiting
// operations to their compare-and-set rules (RUN-R30): only a running Run
// enters waiting, a repeated wait with the same reason changes nothing, only
// a waiting Run leaves it, and only a pending Run records an admission wait.
func TestPostgresWaitingOperationsMoveOnlyAdmittedRuns(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	createTestRun(t, ctx, store, "run-admitted")
	if _, err := admitTestRun(ctx, store, "run-admitted"); err != nil {
		t.Fatal(err)
	}
	createTestRun(t, ctx, store, "run-pending")
	if _, err := store.TransitionRun(ctx, "run-pending", RunInitializing, RunPending, Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	createTestRun(t, ctx, store, "run-cancelling")
	if _, err := admitTestRun(ctx, store, "run-cancelling"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.RequestRunCancellation(ctx, "run-cancelling", WorkflowRunCancellation{
		Code: CancellationUserRequested, RequestedAt: time.Now(),
	}); err != nil {
		t.Fatal(err)
	}
	run := func(runID string) WorkflowRun {
		t.Helper()
		value, err := store.GetRun(ctx, runID)
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	apply := func(operation func(*PostgresTxStore) error) {
		t.Helper()
		if err := inRunStoreTx(ctx, pool, operation); err != nil {
			t.Fatal(err)
		}
	}
	unchanged := func(name string, before WorkflowRun) {
		t.Helper()
		after := run(before.RunID)
		if after.State != before.State || after.StateReason != before.StateReason || !after.UpdatedAt.Equal(before.UpdatedAt) {
			t.Errorf("%s changed Run %s: %s %+v at %s -> %s %+v at %s", name, before.RunID,
				before.State, before.StateReason, before.UpdatedAt, after.State, after.StateReason, after.UpdatedAt)
		}
	}

	apply(func(s *PostgresTxStore) error { return s.EnterWaiting(ctx, "run-admitted", "gateway_unavailable") })
	waiting := run("run-admitted")
	if waiting.State != RunWaiting || waiting.StateReason != (Reason{Code: "gateway_unavailable", Message: gatewayWaitMessage}) {
		t.Fatalf("entered Run = %s %+v, want waiting for the model gateway", waiting.State, waiting.StateReason)
	}
	apply(func(s *PostgresTxStore) error { return s.EnterWaiting(ctx, "run-admitted", "gateway_unavailable") })
	unchanged("a repeated wait with the same reason", waiting)
	apply(func(s *PostgresTxStore) error { return s.EnterWaiting(ctx, "run-admitted", "model_unavailable") })
	if rewaited := run("run-admitted"); rewaited.State != RunWaiting || rewaited.StateReason.Code != "model_unavailable" {
		t.Fatalf("Run waiting for another route failure = %s %+v", rewaited.State, rewaited.StateReason)
	}
	apply(func(s *PostgresTxStore) error { return s.LeaveWaiting(ctx, "run-admitted") })
	running := run("run-admitted")
	if running.State != RunRunning || running.StateReason != (Reason{Code: gatewayRecoveredReason}) {
		t.Fatalf("Run that left waiting = %s %+v, want running after gateway recovery", running.State, running.StateReason)
	}
	apply(func(s *PostgresTxStore) error { return s.LeaveWaiting(ctx, "run-admitted") })
	unchanged("leaving waiting from running", running)
	apply(func(s *PostgresTxStore) error {
		return s.RecordAdmissionWait(ctx, "run-admitted", "gateway_unavailable")
	})
	unchanged("an admission wait of a running Run", running)

	pending := run("run-pending")
	apply(func(s *PostgresTxStore) error { return s.EnterWaiting(ctx, "run-pending", "gateway_unavailable") })
	unchanged("a wait of a pending Run", pending)
	apply(func(s *PostgresTxStore) error { return s.LeaveWaiting(ctx, "run-pending") })
	unchanged("leaving waiting from pending", pending)
	apply(func(s *PostgresTxStore) error {
		return s.RecordAdmissionWait(ctx, "run-pending", "gateway_unavailable")
	})
	deferred := run("run-pending")
	if deferred.State != RunPending || deferred.StateReason != (Reason{Code: "gateway_unavailable", Message: gatewayWaitMessage}) {
		t.Fatalf("pending Run with an admission wait = %s %+v", deferred.State, deferred.StateReason)
	}
	apply(func(s *PostgresTxStore) error {
		return s.RecordAdmissionWait(ctx, "run-pending", "gateway_unavailable")
	})
	unchanged("a repeated admission wait with the same reason", deferred)

	cancelling := run("run-cancelling")
	apply(func(s *PostgresTxStore) error { return s.EnterWaiting(ctx, "run-cancelling", "gateway_unavailable") })
	unchanged("a wait of a cancelling Run", cancelling)
	apply(func(s *PostgresTxStore) error {
		return s.RecordAdmissionWait(ctx, "run-cancelling", "gateway_unavailable")
	})
	unchanged("an admission wait of a cancelling Run", cancelling)

	for name, operation := range map[string]func(*PostgresTxStore) error{
		"wait without a reason":           func(s *PostgresTxStore) error { return s.EnterWaiting(ctx, "run-admitted", " ") },
		"admission wait without a reason": func(s *PostgresTxStore) error { return s.RecordAdmissionWait(ctx, "run-pending", "") },
		"leave without a Run ID":          func(s *PostgresTxStore) error { return s.LeaveWaiting(ctx, "") },
	} {
		if err := inRunStoreTx(ctx, pool, operation); !errors.Is(err, ErrInvalid) {
			t.Errorf("%s = %v, want invalid", name, err)
		}
	}

	// Recovery's port reports a missing Run as no state, which accepts no
	// recovery update.
	err := inRunStoreTx(ctx, pool, func(s *PostgresTxStore) error {
		ports := GatewayRecoveryRuns{}
		if state, err := ports.LockRun(ctx, s.tx, "run-pending"); err != nil || state != string(RunPending) {
			t.Errorf("port lock = (%q, %v), want pending", state, err)
		}
		if state, err := ports.LockRun(ctx, s.tx, "run-missing"); err != nil || state != "" {
			t.Errorf("port lock of a missing Run = (%q, %v), want no state", state, err)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestPostgresFailRunWithActiveStagesEndsEveryActiveStage fails a Run with a
// Stage in every active state and one terminal Stage (SCH-R102): each active
// Stage becomes terminal by its own rule in the Run's transaction, the
// terminal Stage keeps its result, and a Run no longer in the expected state
// changes nothing.
func TestPostgresFailRunWithActiveStagesEndsEveryActiveStage(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	createTestRun(t, ctx, store, "run-failing")
	if _, err := admitTestRun(ctx, store, "run-failing"); err != nil {
		t.Fatal(err)
	}
	candidates := map[string]stageresult.StageContentResult{}
	for _, state := range []StageExecutionState{StagePreparing, StageRunning, StageAborting, StageFinalizing, StageSucceeded} {
		stageID := "stage-" + string(state)
		candidates[stageID] = createStageInState(t, ctx, store, "run-failing", stageID, state)
	}
	params := FailRunWithActiveStagesParams{
		RunID: "run-failing", ExpectedState: RunRunning, NextState: RunFailed,
		Reason:                   Reason{Code: "scheduler_state_invalid"},
		TerminationSchemaVersion: contracts.APIVersion,
		Termination: StageTermination{
			Outcome: TerminationInterrupted, Code: "scheduler_state_invalid",
			Message: "Scheduler could not progress the WorkflowRun", Phase: TerminationRunning, OccurredAt: time.Now(),
		},
		AbortID: "abort-failing",
	}

	stale := params
	stale.ExpectedState = RunWaiting
	if err := inRunStoreTx(ctx, pool, func(s *PostgresTxStore) error {
		return s.FailRunWithActiveStages(ctx, stale)
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("failure from a state the Run is not in = %v, want conflict", err)
	}
	executions, err := store.ListStageExecutions(ctx, "run-failing")
	if err != nil {
		t.Fatal(err)
	}
	for _, execution := range executions {
		if want := StageExecutionState(execution.StageExecutionID[len("stage-"):]); execution.State != want {
			t.Fatalf("a refused failure changed %s to %s", execution.StageExecutionID, execution.State)
		}
	}
	invalid := params
	invalid.NextState = RunSucceeded
	if err := inRunStoreTx(ctx, pool, func(s *PostgresTxStore) error {
		return s.FailRunWithActiveStages(ctx, invalid)
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("failure into succeeded = %v, want invalid", err)
	}

	if err := inRunStoreTx(ctx, pool, func(s *PostgresTxStore) error {
		return s.FailRunWithActiveStages(ctx, params)
	}); err != nil {
		t.Fatal(err)
	}
	failed, err := store.GetRun(ctx, "run-failing")
	if err != nil || failed.State != RunFailed || failed.StateReason.Code != "scheduler_state_invalid" || failed.FinishedAt == nil {
		t.Fatalf("failed Run = (%s %+v, %v)", failed.State, failed.StateReason, err)
	}
	executions, err = store.ListStageExecutions(ctx, "run-failing")
	if err != nil || len(executions) != 5 {
		t.Fatalf("StageExecutions = (%d, %v), want 5", len(executions), err)
	}
	for _, execution := range executions {
		if execution.TerminalAt == nil {
			t.Errorf("%s is %s without terminal_at", execution.StageExecutionID, execution.State)
		}
		switch execution.StageExecutionID {
		case "stage-preparing", "stage-running":
			phase := TerminationPhase(execution.StageExecutionID[len("stage-"):])
			if execution.State != StageInterrupted || execution.StateReason.Code != "termination_committed" ||
				execution.Termination == nil || execution.Termination.Phase != phase ||
				execution.AbortID == nil || *execution.AbortID != "abort-failing" {
				t.Errorf("%s = %s %+v %+v, want interrupted from %s", execution.StageExecutionID,
					execution.State, execution.StateReason, execution.Termination, phase)
			}
		case "stage-aborting":
			if execution.State != StageCancelled || execution.StateReason.Code != "termination_committed" ||
				execution.AbortID == nil || *execution.AbortID != "abort-stage-aborting" {
				t.Errorf("aborting Stage = %s %+v, want its stored cancellation committed", execution.State, execution.StateReason)
			}
		case "stage-finalizing":
			if execution.State != StageSucceeded || execution.StateReason.Code != "result_accepted" ||
				execution.AcceptedResult == nil || execution.AcceptedResult.Summary != candidates["stage-finalizing"].Summary ||
				execution.Termination != nil {
				t.Errorf("finalizing Stage = %s %+v, want its candidate accepted", execution.State, execution.StateReason)
			}
		case "stage-succeeded":
			if execution.State != StageSucceeded || execution.StateReason.Code != "result_accepted" ||
				execution.Termination != nil {
				t.Errorf("terminal Stage = %s %+v, want it unchanged", execution.State, execution.StateReason)
			}
		}
	}
}

// TestPostgresStageAdmissionMarksOnceAndDiscardsOnlyUnadmittedPlacements
// holds MarkStageAdmitted to one admission time (RUN-R28) and
// DiscardUnadmittedAllocations to preparing Stages that were never admitted
// (SCH-R152).
func TestPostgresStageAdmissionMarksOnceAndDiscardsOnlyUnadmittedPlacements(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	createTestRun(t, ctx, store, "run-admission")
	if _, err := store.TransitionRun(ctx, "run-admission", RunInitializing, RunPending, Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	createTestRun(t, ctx, store, "run-admission-other")
	for _, stageID := range []string{"stage-unadmitted", "stage-admitted"} {
		createStageInState(t, ctx, store, "run-admission", stageID, StagePreparing)
		pinned := testStageAllocation(stageID, stageID+"-pinned")
		releasing := testStageAllocation(stageID, stageID+"-releasing")
		releasing.LogicalAgentName, releasing.Namespace = "reviewer", "reviewer"
		for _, allocation := range []StageAllocation{pinned, releasing} {
			if err := store.RecordStageAllocation(ctx, allocation); err != nil {
				t.Fatal(err)
			}
		}
		// A release already attempted is left to terminal release recovery.
		if err := store.MarkStageAllocationReleaseAttempt(ctx, stageID+"-releasing"); err != nil {
			t.Fatal(err)
		}
	}

	admitted, err := store.MarkStageAdmitted(ctx, "stage-admitted")
	if err != nil || admitted.AdmittedAt == nil || admitted.StageExecutionID != "stage-admitted" ||
		admitted.State != StagePreparing {
		t.Fatalf("admitted Stage = (%+v, %v)", admitted, err)
	}
	again, err := store.MarkStageAdmitted(ctx, "stage-admitted")
	if err != nil || again.AdmittedAt == nil || !again.AdmittedAt.Equal(*admitted.AdmittedAt) {
		t.Fatalf("readmitted Stage = (%+v, %v), want the first admission time %s", again.AdmittedAt, err, admitted.AdmittedAt)
	}
	if _, err := store.MarkStageAdmitted(ctx, "stage-missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("admission of a missing Stage = %v, want not found", err)
	}

	discard := func(runID, stageID string) error {
		return inRunStoreTx(ctx, pool, func(s *PostgresTxStore) error {
			if _, err := s.LockRunState(ctx, "run-admission"); err != nil {
				return err
			}
			return s.DiscardUnadmittedAllocations(ctx, runID, stageID)
		})
	}
	allocations := func(stageID string) []string {
		t.Helper()
		recorded, err := store.ListStageAllocations(ctx, stageID)
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]string, 0, len(recorded))
		for _, allocation := range recorded {
			ids = append(ids, allocation.AllocationID)
		}
		return ids
	}
	if err := discard("run-admission", "stage-admitted"); !errors.Is(err, ErrConflict) {
		t.Fatalf("discard of an admitted placement = %v, want conflict", err)
	}
	if got := allocations("stage-admitted"); len(got) != 2 {
		t.Fatalf("a refused discard left allocations %v, want both", got)
	}
	if err := discard("run-admission-other", "stage-unadmitted"); !errors.Is(err, ErrConflict) {
		t.Fatalf("discard under another Run = %v, want conflict", err)
	}
	if err := discard("run-admission", "stage-missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("discard of a missing Stage = %v, want not found", err)
	}
	if err := discard("run-admission", "stage-unadmitted"); err != nil {
		t.Fatal(err)
	}
	if got := allocations("stage-unadmitted"); len(got) != 1 || got[0] != "stage-unadmitted-releasing" {
		t.Fatalf("allocations after the discard = %v, want only the one whose release was attempted", got)
	}
}
