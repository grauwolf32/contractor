package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestPostgresStartPlannerLocksTheRunBeforeTheStage pins RUN-R29 for Planner
// start on both kinds of store. While a Run-first owner holds the Run row,
// StartPlanner waits for that row without owning the Stage row, so the owner
// can still lock the Stage; once the owner commits, the start completes. A
// start that locked the Stage first would hold it here, and the owner's
// Stage lock would close a deadlock cycle.
func TestPostgresStartPlannerLocksTheRunBeforeTheStage(t *testing.T) {
	for _, owner := range []string{"pool", "caller transaction"} {
		t.Run(owner, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			pool := isolatedRunStorePool(t, ctx)
			store := NewPostgresStore(pool)
			const runID, stageID = "run-start-order", "stage-start-order"
			createPreparingStage(t, ctx, store, runID, stageID)
			before := readStartPlannerEffects(t, ctx, pool, stageID)

			holder, holderTx := holdRunLock(t, ctx, pool, runID)
			starter := store
			var caller pgx.Tx
			if owner == "caller transaction" {
				var err error
				caller, err = pool.Begin(ctx)
				if err != nil {
					t.Fatal(err)
				}
				defer func() { _ = caller.Rollback(context.WithoutCancel(ctx)) }()
				starter = NewPostgresStore(caller)
			}
			params := testStartPlannerParams(t, stageID, "session-order")
			done := make(chan error, 1)
			go func() { done <- starter.StartPlanner(ctx, params) }()

			waitForRunRowWaiters(t, ctx, pool, 1, done)
			requireStageRowUnlocked(t, ctx, pool, stageID)
			if _, err := holder.LockStageState(ctx, runID, stageID, StagePreparing); err != nil {
				t.Fatalf("Run-first owner could not lock the Stage while StartPlanner waited: %v", err)
			}
			select {
			case err := <-done:
				t.Fatalf("StartPlanner finished while the Run-first owner held both rows: %v", err)
			default:
			}
			if err := holderTx.Commit(ctx); err != nil {
				t.Fatal(err)
			}
			if err := <-done; err != nil {
				t.Fatalf("StartPlanner after the Run-first owner committed: %v", err)
			}
			if caller != nil {
				if err := caller.Commit(ctx); err != nil {
					t.Fatal(err)
				}
			}
			requireStarted(t, readStartPlannerEffects(t, ctx, pool, stageID), before, params.SessionID)
		})
	}
}

// TestPostgresStartPlannerInACallerTransactionUsesOnlyThatTransaction runs
// StartPlanner in a caller's transaction on a one-connection pool. It joins
// that transaction without acquiring a connection, commits nothing itself,
// and the caller's rollback or commit decides every effect.
func TestPostgresStartPlannerInACallerTransactionUsesOnlyThatTransaction(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	const stageID = "stage-start-caller"
	createPreparingStage(t, ctx, store, "run-start-caller", stageID)
	single := singleConnectionPool(t, ctx, pool)

	for _, finish := range []string{"rollback", "commit"} {
		before := readStartPlannerEffects(t, ctx, pool, stageID)
		tx, err := single.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		acquired := single.Stat().AcquireCount()
		params := testStartPlannerParams(t, stageID, "session-"+finish)
		if err := NewPostgresStore(tx).StartPlanner(ctx, params); err != nil {
			_ = tx.Rollback(ctx)
			t.Fatalf("%s: StartPlanner in the caller's transaction: %v", finish, err)
		}
		if got := single.Stat().AcquireCount(); got != acquired {
			_ = tx.Rollback(ctx)
			t.Fatalf("%s: StartPlanner acquired %d connections beside the caller's transaction", finish, got-acquired)
		}
		inside := readStartPlannerEffects(t, ctx, tx, stageID)
		requireStarted(t, inside, before, params.SessionID)
		if outside := readStartPlannerEffects(t, ctx, pool, stageID); outside != before {
			_ = tx.Rollback(ctx)
			t.Fatalf("%s: StartPlanner committed %+v before its caller, want %+v", finish, outside, before)
		}
		if finish == "rollback" {
			if err := tx.Rollback(ctx); err != nil {
				t.Fatal(err)
			}
			if after := readStartPlannerEffects(t, ctx, pool, stageID); after != before {
				t.Fatalf("effects after the caller's rollback = %+v, want %+v", after, before)
			}
			continue
		}
		if err := tx.Commit(ctx); err != nil {
			t.Fatal(err)
		}
		if after := readStartPlannerEffects(t, ctx, pool, stageID); after != inside {
			t.Fatalf("effects after the caller's commit = %+v, want %+v", after, inside)
		}
	}
}

// TestPostgresStartPlannerLeavesTransactionConflictsToTheirOwner injects a
// definite deadlock abort into the Planner session insert. A caller's
// transaction gets the conflict after one attempt and keeps the retry; on a
// pool StartPlanner replays it under the shared bounded policy with the same
// identities, and an exhausted replay stays a transaction conflict. No
// aborted attempt leaves a session, an event or a consumed sequence number.
func TestPostgresStartPlannerLeavesTransactionConflictsToTheirOwner(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	injection := injectStartPlannerConflicts(t, ctx, pool)

	t.Run("caller transaction", func(t *testing.T) {
		const stageID = "stage-conflict-caller"
		createPreparingStage(t, ctx, store, "run-conflict-caller", stageID)
		before := readStartPlannerEffects(t, ctx, pool, stageID)
		injection.fail(1)
		tx, err := singleConnectionPool(t, ctx, pool).Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		err = NewPostgresStore(tx).StartPlanner(ctx, testStartPlannerParams(t, stageID, "session-caller"))
		if rollbackErr := tx.Rollback(ctx); rollbackErr != nil {
			t.Fatal(rollbackErr)
		}
		if !persistencepostgres.IsTransactionConflict(err) || errors.Is(err, ErrConflict) || injection.attempts() != 1 {
			t.Fatalf("caller-transaction StartPlanner = %v after %d attempts, want one definite transaction conflict",
				err, injection.attempts())
		}
		if after := readStartPlannerEffects(t, ctx, pool, stageID); after != before {
			t.Fatalf("effects after the caller's rollback = %+v, want %+v", after, before)
		}
	})

	t.Run("pool replay", func(t *testing.T) {
		const runID, stageID = "run-conflict-replay", "stage-conflict-replay"
		createPreparingStage(t, ctx, store, runID, stageID)
		before := readStartPlannerEffects(t, ctx, pool, stageID)
		injection.fail(1)
		params := testStartPlannerParams(t, stageID, "session-replay")
		if err := store.StartPlanner(ctx, params); err != nil || injection.attempts() != 2 {
			t.Fatalf("pool StartPlanner = %v after %d attempts, want success on the replay", err, injection.attempts())
		}
		requireStarted(t, readStartPlannerEffects(t, ctx, pool, stageID), before, params.SessionID)
		page, err := store.ListRunEvents(ctx, runID, before.nextEvent-1, 10)
		events := page.Events
		if err != nil || len(events) != 2 || page.More || events[0].Kind != RunEventLifecycleChanged ||
			events[1].EventID != params.RunEvent.EventID || events[1].SequenceNumber != before.nextEvent+1 {
			t.Fatalf("Planner start events = (%+v, %v), want the lifecycle event and %q", events, err, params.RunEvent.EventID)
		}
	})

	t.Run("pool exhausted", func(t *testing.T) {
		policyAttempts := 0
		err := persistencepostgres.InTxWithRetry(ctx, pool, pgx.TxOptions{}, func(pgx.Tx) error {
			policyAttempts++
			return &pgconn.PgError{Code: persistencepostgres.SQLStateDeadlockDetected}
		})
		if !persistencepostgres.IsTransactionConflict(err) || policyAttempts < 2 {
			t.Fatalf("shared retry policy = %v after %d attempts", err, policyAttempts)
		}
		const stageID = "stage-conflict-exhausted"
		createPreparingStage(t, ctx, store, "run-conflict-exhausted", stageID)
		before := readStartPlannerEffects(t, ctx, pool, stageID)
		injection.fail(1 << 30)
		err = store.StartPlanner(ctx, testStartPlannerParams(t, stageID, "session-exhausted"))
		if !persistencepostgres.IsTransactionConflict(err) || errors.Is(err, ErrConflict) ||
			injection.attempts() != int64(policyAttempts) {
			t.Fatalf("pool StartPlanner = %v after %d attempts, want the conflict after the policy's %d",
				err, injection.attempts(), policyAttempts)
		}
		if after := readStartPlannerEffects(t, ctx, pool, stageID); after != before {
			t.Fatalf("effects after exhausted replays = %+v, want %+v", after, before)
		}
	})
}

// TestPostgresStartPlannerConflictsLeaveNoEffects keeps the existing
// classification of a missing or no longer preparing Stage and of a reused
// Planner identity, with no session, event or sequence number left behind.
// Losing the Stage CAS inside a caller's transaction leaves that transaction
// usable.
func TestPostgresStartPlannerConflictsLeaveNoEffects(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	const stageID, otherStageID = "stage-conflict", "stage-conflict-other"
	createPreparingStage(t, ctx, store, "run-conflict", stageID)
	createPreparingStage(t, ctx, store, "run-conflict-other", otherStageID)

	initial := readStartPlannerEffects(t, ctx, pool, stageID)
	var stateConflict *StateConflictError
	err := store.StartPlanner(ctx, testStartPlannerParams(t, "stage-missing", "session-missing"))
	if !errors.As(err, &stateConflict) || stateConflict.Expected != string(StagePreparing) {
		t.Fatalf("missing Stage StartPlanner = %v, want the preparing state conflict", err)
	}
	if after := readStartPlannerEffects(t, ctx, pool, stageID); after != initial {
		t.Fatalf("effects after a missing Stage = %+v, want %+v", after, initial)
	}

	first := testStartPlannerParams(t, stageID, "session-first")
	if err := store.StartPlanner(ctx, first); err != nil {
		t.Fatal(err)
	}
	started := readStartPlannerEffects(t, ctx, pool, stageID)
	requireStarted(t, started, initial, first.SessionID)
	err = store.StartPlanner(ctx, testStartPlannerParams(t, stageID, "session-second"))
	if !errors.As(err, &stateConflict) || stateConflict.ID != stageID {
		t.Fatalf("second StartPlanner = %v, want the preparing state conflict", err)
	}

	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	err = NewPostgresStore(tx).StartPlanner(ctx, testStartPlannerParams(t, stageID, "session-caller"))
	if !errors.As(err, &stateConflict) {
		t.Fatalf("caller-transaction StartPlanner of a running Stage = %v, want the state conflict", err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("caller's transaction after a lost Stage CAS: %v", err)
	}
	if after := readStartPlannerEffects(t, ctx, pool, stageID); after != started {
		t.Fatalf("effects after lost Stage CAS = %+v, want %+v", after, started)
	}

	otherBefore := readStartPlannerEffects(t, ctx, pool, otherStageID)
	reused := testStartPlannerParams(t, otherStageID, first.SessionID)
	if err := store.StartPlanner(ctx, reused); !errors.Is(err, ErrConflict) || errors.As(err, &stateConflict) {
		t.Fatalf("StartPlanner with a reused session ID = %v, want the identity conflict", err)
	}
	if after := readStartPlannerEffects(t, ctx, pool, otherStageID); after != otherBefore {
		t.Fatalf("effects after a reused session ID = %+v, want %+v", after, otherBefore)
	}
}

func TestPostgresStartPlannerRejectsCancelledRun(t *testing.T) {
	for _, waiting := range []bool{false, true} {
		t.Run(fmt.Sprintf("waiting=%t", waiting), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool := isolatedRunStorePool(t, ctx)
			store := NewPostgresStore(pool)
			const runID, stageID = "run-cancel-start", "stage-cancel-start"
			createPreparingStage(t, ctx, store, runID, stageID)
			before := readStartPlannerEffects(t, ctx, pool, stageID)
			request := WorkflowRunCancellation{Code: CancellationUserRequested, RequestedAt: time.Now()}
			var started <-chan error
			if waiting {
				locked, tx := holdRunLock(t, ctx, pool, runID)
				if _, err := locked.RequestRunCancellation(ctx, runID, request); err != nil {
					t.Fatal(err)
				}
				done := make(chan error, 1)
				go func() { done <- store.StartPlanner(ctx, testStartPlannerParams(t, stageID, "session-cancel-start")) }()
				waitForRunRowWaiters(t, ctx, pool, 1, done)
				if err := tx.Commit(ctx); err != nil {
					t.Fatal(err)
				}
				started = done
			} else {
				if _, err := store.RequestRunCancellation(ctx, runID, request); err != nil {
					t.Fatal(err)
				}
				done := make(chan error, 1)
				done <- store.StartPlanner(ctx, testStartPlannerParams(t, stageID, "session-cancel-start"))
				started = done
			}
			if err := <-started; !errors.Is(err, ErrConflict) {
				t.Fatalf("Planner start after cancellation = %v, want ErrConflict", err)
			}
			after := readStartPlannerEffects(t, ctx, pool, stageID)
			// Cancellation appends exactly its own lifecycle event.
			before.events++
			before.nextEvent++
			if after != before {
				t.Fatalf("Planner start changed cancelled Run: %+v, want %+v", after, before)
			}
		})
	}
}

// startPlannerEffects is everything StartPlanner changes for one Stage: its
// state and Planner session, the Planner sessions of the Stage, and the
// owning Run's events and next event sequence.
type startPlannerEffects struct {
	state     StageExecutionState
	sessionID string
	sessions  int
	events    int
	nextEvent int64
}

func readStartPlannerEffects(
	t *testing.T, ctx context.Context, db persistencepostgres.DBTX, stageID string,
) startPlannerEffects {
	t.Helper()
	var effects startPlannerEffects
	if err := db.QueryRow(ctx, `
SELECT stage.state, COALESCE(stage.planner_session_id, ''),
       (SELECT count(*) FROM planner_sessions WHERE stage_execution_id = stage.stage_execution_id),
       (SELECT count(*) FROM workflow_run_events WHERE run_id = run.run_id),
       run.next_run_event_sequence
FROM stage_executions AS stage JOIN workflow_runs AS run ON run.run_id = stage.run_id
WHERE stage.stage_execution_id = $1`, stageID).Scan(
		&effects.state, &effects.sessionID, &effects.sessions, &effects.events, &effects.nextEvent,
	); err != nil {
		t.Fatalf("read StartPlanner effects of %q: %v", stageID, err)
	}
	return effects
}

// requireStarted checks one complete start after before: the Stage runs
// session sessionID, which is its only one, and the Run gained exactly the
// lifecycle and Planner started events with the next two sequence numbers.
func requireStarted(t *testing.T, got, before startPlannerEffects, sessionID string) {
	t.Helper()
	want := startPlannerEffects{
		state: StageRunning, sessionID: sessionID, sessions: 1,
		events: before.events + 2, nextEvent: before.nextEvent + 2,
	}
	if got != want {
		t.Fatalf("StartPlanner effects = %+v, want %+v", got, want)
	}
}

func createPreparingStage(t *testing.T, ctx context.Context, store *PostgresStore, runID, stageID string) {
	t.Helper()
	createTestRun(t, ctx, store, runID)
	if _, err := admitTestRun(ctx, store, runID); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateStageExecution(ctx, CreateStageExecutionParams{
		LaneClass:        resolved.LaneClassModel,
		StageExecutionID: stageID, RunID: runID, StageName: "copy", Attempt: 1,
		StageSpecSchemaVersion:    contracts.APIVersion,
		StageSpecSnapshot:         json.RawMessage(`{"objective":"copy"}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext:              StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]PinnedContextArtifact{}},
	}); err != nil {
		t.Fatal(err)
	}
}

func testStartPlannerParams(t *testing.T, stageID, sessionID string) StartPlannerParams {
	t.Helper()
	identity := plannerRunEventData{
		StageExecutionID: stageID, SessionID: sessionID, InvocationID: sessionID + "-invocation",
	}
	return StartPlannerParams{
		StageExecutionID: stageID, SessionID: sessionID, InvocationID: identity.InvocationID,
		StateSchemaVersion: contracts.APIVersion, InitialState: json.RawMessage(`{"step":0}`),
		RunEvent: RunEventAppend{
			EventID: sessionID + "-started", EventSchemaVersion: contracts.APIVersion,
			Kind: RunEventPlannerStarted, Data: mustRunEventJSON(t, identity),
		},
		Reason: Reason{Code: "planner_started"},
	}
}

// holdRunLock begins a transaction that locks Run runID first, as every
// Run-first owner does (LockRunState). The test commits it or its cleanup
// rolls it back.
func holdRunLock(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string) (*PostgresTxStore, pgx.Tx) {
	t.Helper()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Rollback(context.WithoutCancel(ctx)) })
	holder := NewPostgresTxStore(tx)
	if _, err := holder.LockRunState(ctx, runID); err != nil {
		t.Fatal(err)
	}
	return holder, tx
}

// waitForRunRowWaiters waits until want backends wait for a workflow_runs
// row lock: the first holds the row's tuple lock while it waits for the
// owner, later ones queue for that tuple lock. It fails at once if a backend
// waits for a stage_executions row instead, which only a start that locked
// its Stage first can cause, and if an operation reports on done first; the
// context deadline is only a hang guard.
func waitForRunRowWaiters(t *testing.T, ctx context.Context, pool *pgxpool.Pool, want int, done <-chan error) {
	t.Helper()
	for {
		var waiting, stageWaiting int
		if err := pool.QueryRow(ctx, waitingRowLockersSQL).Scan(&waiting, &stageWaiting); err != nil {
			t.Fatal(err)
		}
		if stageWaiting != 0 {
			t.Fatalf("%d backends wait for a Stage row behind a start that locked it first (RUN-R29)", stageWaiting)
		}
		if waiting == want {
			return
		}
		select {
		case err := <-done:
			t.Fatalf("operation finished before %d backends waited for the Run row: %v", want, err)
		case <-ctx.Done():
			t.Fatalf("%d of %d backends wait for the Run row: %v", waiting, want, ctx.Err())
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// waitingRowLockersSQL counts the backends that wait for a row lock of
// workflow_runs and of stage_executions. A row lock waiter holds or queues
// for the row's tuple lock.
const waitingRowLockersSQL = `
SELECT count(DISTINCT pid) FILTER (WHERE relation = 'workflow_runs'::regclass),
       count(DISTINCT pid) FILTER (WHERE relation = 'stage_executions'::regclass)
FROM pg_locks
WHERE locktype = 'tuple' AND cardinality(pg_blocking_pids(pid)) > 0`

// requireStageRowUnlocked fails unless another transaction can lock Stage
// stageID's row at once, which it cannot while a waiting start owns it.
func requireStageRowUnlocked(t *testing.T, ctx context.Context, pool *pgxpool.Pool, stageID string) {
	t.Helper()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if _, err := tx.Exec(ctx, `SELECT 1 FROM stage_executions WHERE stage_execution_id = $1 FOR UPDATE NOWAIT`, stageID); err != nil {
		t.Fatalf("StartPlanner owns the Stage row while it waits for the Run row (RUN-R29): %v", err)
	}
}

func singleConnectionPool(t *testing.T, ctx context.Context, pool *pgxpool.Pool) *pgxpool.Pool {
	t.Helper()
	config := pool.Config()
	config.MaxConns, config.MinConns = 1, 0
	single, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(single.Close)
	return single
}

// startPlannerConflictInjection makes Planner session inserts fail with a
// definite deadlock abort. A sequence counts the attempts, because a
// sequence advance survives the rollback of the attempt that made it.
type startPlannerConflictInjection struct {
	t    *testing.T
	ctx  context.Context
	pool *pgxpool.Pool
}

func injectStartPlannerConflicts(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
) startPlannerConflictInjection {
	t.Helper()
	for _, statement := range []string{
		`CREATE SEQUENCE start_planner_attempts`,
		`CREATE TABLE start_planner_failures (failures bigint NOT NULL)`,
		`INSERT INTO start_planner_failures VALUES (0)`,
		`CREATE FUNCTION inject_start_planner_conflict() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF nextval('start_planner_attempts') <= (SELECT failures FROM start_planner_failures) THEN
        RAISE EXCEPTION 'injected Planner start conflict' USING ERRCODE = 'deadlock_detected';
    END IF;
    RETURN NEW;
END $$`,
		`CREATE TRIGGER inject_start_planner_conflict BEFORE INSERT ON planner_sessions
FOR EACH ROW EXECUTE FUNCTION inject_start_planner_conflict()`,
	} {
		if _, err := pool.Exec(ctx, statement); err != nil {
			t.Fatal(err)
		}
	}
	return startPlannerConflictInjection{t: t, ctx: ctx, pool: pool}
}

// fail resets the attempt count and fails the next failures attempts.
func (i startPlannerConflictInjection) fail(failures int64) {
	i.t.Helper()
	if _, err := i.pool.Exec(i.ctx, `
WITH reset AS (SELECT setval('start_planner_attempts', 1, false))
UPDATE start_planner_failures SET failures = $1 FROM reset`, failures); err != nil {
		i.t.Fatal(err)
	}
}

func (i startPlannerConflictInjection) attempts() int64 {
	i.t.Helper()
	var attempts int64
	if err := i.pool.QueryRow(i.ctx, `
SELECT CASE WHEN is_called THEN last_value ELSE 0 END FROM start_planner_attempts`).Scan(&attempts); err != nil {
		i.t.Fatal(err)
	}
	return attempts
}
