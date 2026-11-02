package session

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	lockOrderRunID   = "run-planner"
	lockOrderStageID = "stage-planner"
)

// beginResult is what one concurrent Service.Begin returned.
type beginResult struct {
	start planner.SessionStart
	err   error
}

// TestPostgresBeginLocksTheRunBeforeTheStage pins RUN-R29 on the production
// Planner start path. While a Run-first owner holds the Run row, Begin waits
// for that row without owning the Stage row; the owner then locks the Stage
// as every Run-first commit point does, and the start completes after the
// owner commits. A start that locked the Stage before the Run would hold it
// here and close a deadlock cycle with the owner.
func TestPostgresBeginLocksTheRunBeforeTheStage(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool, store, service := lockOrderFixture(t, ctx)
	before := readBeginEffects(t, ctx, pool)

	holder, holderTx := holdPlannerRunLock(t, ctx, pool)
	done := make(chan beginResult, 1)
	go func() {
		start, err := service.Begin(ctx, lockOrderStageID)
		done <- beginResult{start: start, err: err}
	}()
	waitForPlannerRunWaiters(t, ctx, pool, 1, done)
	requirePlannerStageUnlocked(t, ctx, pool)
	if _, err := holder.LockStageState(ctx, lockOrderRunID, lockOrderStageID, runstore.StagePreparing); err != nil {
		t.Fatalf("Run-first owner could not lock the Stage while Begin waited: %v", err)
	}
	select {
	case result := <-done:
		t.Fatalf("Begin finished while the Run-first owner held both rows: %+v", result)
	default:
	}
	if err := holderTx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	result := <-done
	if result.err != nil || !result.start.Invoke {
		t.Fatalf("Begin after the Run-first owner committed = %+v", result)
	}
	requireBeginStarted(t, ctx, pool, store, before, result.start.Identity)
}

// TestPostgresConcurrentBeginsStartOneSession releases two Begin calls that
// both wait for the Run row. One creates the session and its two events; the
// other loses the Stage CAS, reloads the running Stage and reports the
// invocation in progress, as before.
func TestPostgresConcurrentBeginsStartOneSession(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool, store, service := lockOrderFixture(t, ctx)
	before := readBeginEffects(t, ctx, pool)

	_, holderTx := holdPlannerRunLock(t, ctx, pool)
	done := make(chan beginResult, 2)
	for range 2 {
		go func() {
			start, err := service.Begin(ctx, lockOrderStageID)
			done <- beginResult{start: start, err: err}
		}()
	}
	waitForPlannerRunWaiters(t, ctx, pool, 2, done)
	requirePlannerStageUnlocked(t, ctx, pool)
	if err := holderTx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	var winner *planner.SessionIdentity
	inProgress := 0
	for range 2 {
		result := <-done
		switch {
		case result.err == nil && result.start.Invoke && winner == nil:
			winner = &result.start.Identity
		case errors.Is(result.err, planner.ErrInvocationInProgress):
			inProgress++
		default:
			t.Fatalf("concurrent Begin = %+v", result)
		}
	}
	if winner == nil || inProgress != 1 {
		t.Fatalf("concurrent Begins started %v with %d in progress, want one start", winner, inProgress)
	}
	requireBeginStarted(t, ctx, pool, store, before, *winner)
}

// TestPostgresBeginAfterARunFirstAbortLeavesNoStartEffects aborts the
// preparing Stage in a Run-first transaction, the Scheduler's lock order,
// while Begin waits for the Run row. Begin then loses the Stage CAS and fails
// on the reloaded Stage with no session, no Planner event and no consumed
// sequence number; only the abort's own lifecycle event is appended.
func TestPostgresBeginAfterARunFirstAbortLeavesNoStartEffects(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool, _, service := lockOrderFixture(t, ctx)
	before := readBeginEffects(t, ctx, pool)

	holder, holderTx := holdPlannerRunLock(t, ctx, pool)
	done := make(chan beginResult, 1)
	go func() {
		start, err := service.Begin(ctx, lockOrderStageID)
		done <- beginResult{start: start, err: err}
	}()
	waitForPlannerRunWaiters(t, ctx, pool, 1, done)
	requirePlannerStageUnlocked(t, ctx, pool)
	if _, err := holder.LockStageState(ctx, lockOrderRunID, lockOrderStageID, runstore.StagePreparing); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	if err := holder.EnterAborting(ctx, runstore.EnterAbortingParams{
		StageExecutionID: lockOrderStageID, ExpectedState: runstore.StagePreparing,
		TerminationSchemaVersion: contracts.APIVersion,
		Termination: runstore.StageTermination{
			Outcome: runstore.TerminationCancelled, Code: "run_cancelled", Message: "Run cancelled",
			Phase: runstore.TerminationPreparing, OccurredAt: now,
		},
		AbortID: "abort-preparing", Deadline: now.Add(time.Minute), Reason: runstore.Reason{Code: "run_cancelled"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := holderTx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	result := <-done
	if result.err == nil || !strings.Contains(result.err.Error(), "no recoverable Planner session") {
		t.Fatalf("Begin after the abort = %+v, want the reloaded Stage without a session", result)
	}
	want := beginEffects{
		state: runstore.StageAborting, sessions: 0, events: before.events + 1, nextEvent: before.nextEvent + 1,
	}
	if got := readBeginEffects(t, ctx, pool); got != want {
		t.Fatalf("effects after the abort = %+v, want %+v", got, want)
	}
}

// TestPostgresCancelledBeginLeavesNoStartEffects cancels Begin while it waits
// for the Run row. The cancellation is reported as such, the start leaves no
// effect, and a later start allocates the very next sequence numbers.
func TestPostgresCancelledBeginLeavesNoStartEffects(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool, store, service := lockOrderFixture(t, ctx)
	before := readBeginEffects(t, ctx, pool)

	_, holderTx := holdPlannerRunLock(t, ctx, pool)
	beginCtx, cancelBegin := context.WithCancel(ctx)
	defer cancelBegin()
	done := make(chan beginResult, 1)
	go func() {
		start, err := service.Begin(beginCtx, lockOrderStageID)
		done <- beginResult{start: start, err: err}
	}()
	waitForPlannerRunWaiters(t, ctx, pool, 1, done)
	requirePlannerStageUnlocked(t, ctx, pool)
	cancelBegin()
	result := <-done
	if !errors.Is(result.err, context.Canceled) {
		t.Fatalf("cancelled Begin = %+v, want context cancellation", result)
	}
	if err := holderTx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if got := readBeginEffects(t, ctx, pool); got != before {
		t.Fatalf("effects after a cancelled Begin = %+v, want %+v", got, before)
	}
	started, err := service.Begin(ctx, lockOrderStageID)
	if err != nil || !started.Invoke {
		t.Fatalf("Begin after the cancelled one = (%+v, %v)", started, err)
	}
	requireBeginStarted(t, ctx, pool, store, before, started.Identity)
}

func lockOrderFixture(t *testing.T, ctx context.Context) (*pgxpool.Pool, *runstore.PostgresStore, *Service) {
	t.Helper()
	pool := isolatedPlannerPool(t, ctx)
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createPlannerStage(t, ctx, store)
	service, err := New(store, Options{})
	if err != nil {
		t.Fatal(err)
	}
	return pool, store, service
}

// beginEffects is everything a Planner start changes: the Stage state and
// Planner session, the sessions of the Stage, and the Run's events and next
// event sequence.
type beginEffects struct {
	state     runstore.StageExecutionState
	sessionID string
	sessions  int
	events    int
	nextEvent int64
}

func readBeginEffects(t *testing.T, ctx context.Context, pool *pgxpool.Pool) beginEffects {
	t.Helper()
	var effects beginEffects
	if err := pool.QueryRow(ctx, `
SELECT stage.state, COALESCE(stage.planner_session_id, ''),
       (SELECT count(*) FROM planner_sessions WHERE stage_execution_id = stage.stage_execution_id),
       (SELECT count(*) FROM workflow_run_events WHERE run_id = run.run_id),
       run.next_run_event_sequence
FROM stage_executions AS stage JOIN workflow_runs AS run ON run.run_id = stage.run_id
WHERE stage.stage_execution_id = $1`, lockOrderStageID).Scan(
		&effects.state, &effects.sessionID, &effects.sessions, &effects.events, &effects.nextEvent,
	); err != nil {
		t.Fatal(err)
	}
	return effects
}

// requireBeginStarted checks one complete start after before: the Stage runs
// the started session, which is its only one, and the Run gained exactly its
// lifecycle and Planner started events at the next two sequence numbers.
func requireBeginStarted(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, store *runstore.PostgresStore,
	before beginEffects, identity planner.SessionIdentity,
) {
	t.Helper()
	want := beginEffects{
		state: runstore.StageRunning, sessionID: identity.SessionID, sessions: 1,
		events: before.events + 2, nextEvent: before.nextEvent + 2,
	}
	if got := readBeginEffects(t, ctx, pool); got != want {
		t.Fatalf("Planner start effects = %+v, want %+v", got, want)
	}
	page, err := store.ListRunEvents(ctx, lockOrderRunID, before.nextEvent-1, 10)
	events := page.Events
	if err != nil || len(events) != 2 || page.More || events[0].Kind != runstore.RunEventLifecycleChanged ||
		events[0].SequenceNumber != before.nextEvent || events[1].Kind != runstore.RunEventPlannerStarted ||
		!strings.Contains(string(events[1].Data), identity.SessionID) {
		t.Fatalf("Planner start events = (%+v, %v)", events, err)
	}
}

// holdPlannerRunLock begins a transaction that locks the Run first, as every
// Run-first owner does. The test commits it or its cleanup rolls it back.
func holdPlannerRunLock(t *testing.T, ctx context.Context, pool *pgxpool.Pool) (*runstore.PostgresTxStore, pgx.Tx) {
	t.Helper()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Rollback(context.WithoutCancel(ctx)) })
	holder := runstore.NewPostgresTxStore(tx, artifactpgtest.NewStorage())
	if _, err := holder.LockRunState(ctx, lockOrderRunID); err != nil {
		t.Fatal(err)
	}
	return holder, tx
}

// waitForPlannerRunWaiters waits until want backends wait for a
// workflow_runs row lock: the first holds the row's tuple lock while it waits
// for the owner, later ones queue for that tuple lock. It fails at once if a
// backend waits for a stage_executions row instead, which only a start that
// locked its Stage first can cause, and if a Begin returns first; the context
// deadline is only a hang guard.
func waitForPlannerRunWaiters(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, want int, done <-chan beginResult,
) {
	t.Helper()
	for {
		var waiting, stageWaiting int
		if err := pool.QueryRow(ctx, `
SELECT count(DISTINCT pid) FILTER (WHERE relation = 'workflow_runs'::regclass),
       count(DISTINCT pid) FILTER (WHERE relation = 'stage_executions'::regclass)
FROM pg_locks
WHERE locktype = 'tuple' AND cardinality(pg_blocking_pids(pid)) > 0`).Scan(&waiting, &stageWaiting); err != nil {
			t.Fatal(err)
		}
		if stageWaiting != 0 {
			t.Fatalf("%d starts wait for the Stage row behind a start that locked it first (RUN-R29)", stageWaiting)
		}
		if waiting == want {
			return
		}
		select {
		case result := <-done:
			t.Fatalf("Begin finished before %d starts waited for the Run row: %+v", want, result)
		case <-ctx.Done():
			t.Fatalf("%d of %d starts wait for the Run row: %v", waiting, want, ctx.Err())
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// requirePlannerStageUnlocked fails unless another transaction can lock the
// Stage row at once, which it cannot while a waiting start owns it.
func requirePlannerStageUnlocked(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if _, err := tx.Exec(ctx, `SELECT 1 FROM stage_executions WHERE stage_execution_id = $1 FOR UPDATE NOWAIT`,
		lockOrderStageID); err != nil {
		t.Fatalf("Begin owns the Stage row while it waits for the Run row (RUN-R29): %v", err)
	}
}
