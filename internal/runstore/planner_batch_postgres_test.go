package runstore

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type plannerBatchTrace struct{ reads atomic.Int64 }

func (trace *plannerBatchTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	if strings.HasPrefix(data.SQL, "SELECT ") && strings.Contains(data.SQL, "FROM planner_sessions") {
		trace.reads.Add(1)
	}
	return ctx
}

func (*plannerBatchTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func TestPostgresPlannerSessionBatchConsumesRowsAndClosesOnEveryOutcome(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	trace := &plannerBatchTrace{}
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{Configure: func(config *pgxpool.Config) {
		config.MaxConns = 1
		config.ConnConfig.Tracer = trace
	}})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	populateConsolidationRuns(t, ctx, pool)
	const sessions = 200
	if _, err := pool.Exec(ctx, `INSERT INTO planner_sessions (
session_id, stage_execution_id, invocation_id, state_schema_version, state)
SELECT 'session-'||stage_execution_id,stage_execution_id,'invocation-'||stage_execution_id,'contractor/v1alpha1',
jsonb_build_object('padding',repeat('x',32768))
FROM stage_executions ORDER BY stage_execution_id LIMIT $1`, sessions); err != nil {
		t.Fatal(err)
	}
	rows, err := pool.Query(ctx, `SELECT session_id FROM planner_sessions ORDER BY session_id`)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatal(err)
	}
	store := NewPostgresStore(pool)
	trace.reads.Store(0)
	seen := make(map[string]bool, sessions)
	if err := store.ReadPlannerSessions(ctx, append(ids, "missing-session"), func(session PlannerSession) error {
		if seen[session.SessionID] || len(session.State) < 32768 {
			t.Fatalf("duplicate or incomplete session %s", session.SessionID)
		}
		seen[session.SessionID] = true
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(seen) != sessions || trace.reads.Load() != 1 {
		t.Fatalf("consumed %d sessions in %d reads, want %d in one", len(seen), trace.reads.Load(), sessions)
	}
	if err := store.ReadPlannerSessions(ctx, nil, func(PlannerSession) error { t.Fatal("empty batch called consumer"); return nil }); err != nil {
		t.Fatal(err)
	}
	if err := store.ReadPlannerSessions(ctx, ids, nil); !errors.Is(err, ErrInvalid) || trace.reads.Load() != 1 {
		t.Fatalf("invalid/empty batch queried the database: %v, reads=%d", err, trace.reads.Load())
	}
	consumerFailure := errors.New("consumer refused the state")
	if err := store.ReadPlannerSessions(ctx, ids, func(PlannerSession) error { return consumerFailure }); !errors.Is(err, consumerFailure) {
		t.Fatalf("consumer error = %v", err)
	}
	if pool.Stat().AcquiredConns() != 0 {
		t.Fatal("consumer error leaked the only connection")
	}
	cancelled, stop := context.WithCancel(ctx)
	stop()
	if err := store.ReadPlannerSessions(cancelled, ids, func(PlannerSession) error { return nil }); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled read = %v", err)
	}
	// Both readers share the same stored-state shape validation. Damage the
	// stored value past its CHECK, then ensure an early scan failure also
	// releases the single connection.
	unguardTable(t, ctx, pool, "planner_sessions")
	if _, err := pool.Exec(ctx, `UPDATE planner_sessions SET state='[]'::jsonb WHERE session_id=$1`, ids[0]); err != nil {
		t.Fatal(err)
	}
	if _, err := store.GetPlannerSession(ctx, ids[0]); err == nil {
		t.Fatal("single read accepted damaged state")
	}
	if err := store.ReadPlannerSessions(ctx, ids[:1], func(PlannerSession) error { t.Fatal("damaged state reached consumer"); return nil }); err == nil {
		t.Fatal("batch read accepted damaged state")
	}
	if pool.Stat().AcquiredConns() != 0 {
		t.Fatal("scan error leaked the only connection")
	}
}
