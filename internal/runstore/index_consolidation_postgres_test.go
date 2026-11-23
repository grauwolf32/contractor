package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// consolidationSlack is the allowance every budget below adds for index
// entries of row versions that earlier measurements left behind: a measured
// mutation rolls back, but its row versions stay until VACUUM, and the
// rows-read count includes them.
const consolidationSlack = 8

// consolidationRunsSQL loads 1,200 Runs of six owners over four Projects each
// (one in seven without a Project): every 50th Run is active, the others
// alternate between failed and succeeded.
const consolidationRunsSQL = `
INSERT INTO workflow_runs (
    run_id, owner_id, project_id, workflow_name, workflow_version, workflow_schema_version,
    workflow_snapshot, parameters, runtime_labels, runtime_config_snapshot, scheduler_lane_class,
    state, state_reason_code, created_at, updated_at, started_at, finished_at
)
SELECT 'run-' || lpad(i::text, 5, '0'), 'owner-' || (i % 6),
       CASE WHEN i % 7 = 3 THEN NULL ELSE 'project-' || (i % 6) || '-' || ((i / 6) % 4) END,
       'consolidation', '1', 'contractor/v1alpha1', '{}'::jsonb, '{}'::jsonb, '{}'::text[], $1::jsonb, 'model',
       run.state, 'fixture', run.created_at, run.created_at,
       CASE WHEN run.state NOT IN ('initializing', 'pending') THEN run.created_at END,
       CASE WHEN run.state IN ('succeeded', 'failed') THEN run.created_at + interval '1 minute' END
FROM generate_series(1, 1200) AS i
CROSS JOIN LATERAL (
    SELECT timestamptz '2026-09-01 00:00:00+00' + i * interval '1 second' AS created_at,
           CASE
               WHEN i % 50 = 0 THEN (ARRAY['initializing', 'pending', 'running', 'waiting'])[(i / 50) % 4 + 1]
               WHEN i % 2 = 1 THEN 'failed'
               ELSE 'succeeded'
           END AS state
) AS run`

// consolidationStagesSQL gives every Run three StageExecutions. The third
// Stage of a running or waiting Run is active: preparing, or aborting for
// every 100th Run. Every other Stage was interrupted.
const consolidationStagesSQL = `
INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt,
    stage_spec_schema_version, stage_spec_snapshot, stage_context_schema_version, stage_context_snapshot,
    state, state_reason_code, termination_schema_version, stage_termination,
    abort_id, abort_deadline, terminal_at, created_at, updated_at
)
SELECT run.run_id || '-stage-' || stage, run.run_id, 'stage-' || stage, 1,
       'contractor/v1alpha1', '{}'::jsonb, 'contractor/v1alpha1', '{"parameters":{},"artifacts":{}}'::jsonb,
       CASE WHEN live AND aborting THEN 'aborting' WHEN live THEN 'preparing' ELSE 'interrupted' END,
       CASE WHEN live AND aborting THEN 'abort_requested' WHEN live THEN 'created' ELSE 'termination_committed' END,
       CASE WHEN NOT live OR aborting THEN 'contractor/v1alpha1' END,
       CASE WHEN NOT live OR aborting THEN $1::jsonb END,
       CASE WHEN NOT live OR aborting THEN 'abort-' || stage END,
       CASE WHEN NOT live OR aborting THEN run.created_at END,
       CASE WHEN NOT live THEN run.created_at END,
       run.created_at + stage * interval '1 millisecond', run.created_at + stage * interval '1 millisecond'
FROM workflow_runs AS run
CROSS JOIN generate_series(1, 3) AS stage
CROSS JOIN LATERAL (SELECT run.state IN ('running', 'waiting') AND stage = 3 AS live,
                           substring(run.run_id FROM 5)::integer % 100 = 0 AS aborting) AS current`

// consolidationEventsSQL stores 2,000 events for run-00002 and 40 for each
// other of the first 300 Runs; ten consecutive events share occurred_at.
const consolidationEventsSQL = `
INSERT INTO workflow_run_events (run_id, sequence_number, event_id, event_schema_version, kind, data, occurred_at)
SELECT run.run_id, n, run.run_id || '-event-' || n, 'contractor/v1alpha1', 'planner.started',
       jsonb_build_object('n', n), run.created_at + (n / 10) * interval '1 second'
FROM workflow_runs AS run
CROSS JOIN LATERAL generate_series(1, CASE WHEN run.run_id = 'run-00002' THEN 2000 ELSE 40 END) AS n
WHERE run.run_id <= 'run-00300'`

// populateConsolidationRuns loads the Run, Stage and event fixture of
// TestPostgresRunReadersStayBoundedWithoutConsolidatedIndexes. The rows
// satisfy every constraint and foreign key; the user triggers that guard the
// store's own writes are off while they load.
func populateConsolidationRuns(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	snapshot, err := json.Marshal(runtimeconfigtest.BuiltInRunSnapshot())
	if err != nil {
		t.Fatal(err)
	}
	termination := `{"outcome":"interrupted","code":"control_lease_expired","message":"allocation lost",` +
		`"retryable":true,"phase":"preparing","occurredAt":"2026-09-01T00:00:00Z"}`
	err = pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		for _, statement := range []struct {
			sql  string
			args []any
		}{
			{sql: `SET CONSTRAINTS ALL IMMEDIATE`},
			{sql: `ALTER TABLE workflow_runs DISABLE TRIGGER USER`},
			{sql: `ALTER TABLE stage_executions DISABLE TRIGGER USER`},
			{sql: `ALTER TABLE workflow_run_events DISABLE TRIGGER USER`},
			{sql: `INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
SELECT 'project-' || o || '-' || p, 'owner-' || o, 'project', 'Fixture', 'create-' || p, 'sha256:' || repeat('a', 64)
  FROM generate_series(0, 5) AS o, generate_series(0, 3) AS p`},
			{sql: consolidationRunsSQL, args: []any{snapshot}},
			{sql: consolidationStagesSQL, args: []any{termination}},
			{sql: consolidationEventsSQL},
			{sql: `ALTER TABLE workflow_runs ENABLE TRIGGER USER`},
			{sql: `ALTER TABLE stage_executions ENABLE TRIGGER USER`},
			{sql: `ALTER TABLE workflow_run_events ENABLE TRIGGER USER`},
		} {
			if _, err := tx.Exec(ctx, statement.sql, statement.args...); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("populate Runs: %v", err)
	}
	if _, err := pool.Exec(ctx, `VACUUM ANALYZE projects, workflow_runs, stage_executions, workflow_run_events`); err != nil {
		t.Fatal(err)
	}
}

// TestPostgresRunReadersStayBoundedWithoutConsolidatedIndexes reads, in the
// custom and the generic plan, the Run and Stage statements that the indexes
// V409-007 dropped could serve: workflow_run_events_time_idx (run_id,
// occurred_at, sequence_number) and stage_executions_recovery_idx (state,
// updated_at) over the active states. The event page walks one Run's primary
// key, so a page reads the events it returns and one look-ahead, and paging a
// 2,000-event history with repeated occurred_at values returns every event
// once, in sequence order. The active-Stage statements read the Stages of
// their Run through stage_executions_run_active_idx or the Run's identity
// indexes: at most the Run's three Stages. Before the drop the statements
// read the same rows, the page walk through the same primary key, except the
// finalizing-candidate update, which read no Stage through the dropped index
// and now reads the Run's one active Stage.
func TestPostgresRunReadersStayBoundedWithoutConsolidatedIndexes(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	populateConsolidationRuns(t, ctx, pool)

	const pageSize = 100
	for _, history := range []struct {
		runID  string
		events int64
	}{{"run-00002", 2000}, {"run-00150", 40}} {
		for _, mode := range explaintest.Modes {
			var delivered []int64
			pages, largest := 0, int64(0)
			for after, more := int64(0), true; more; pages++ {
				analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, listRunEventsSQL,
					history.runID, after, pageSize, maxRunEventPageDataBytes)
				if err != nil {
					t.Fatalf("%s %s after %d: %v", history.runID, mode, after, err)
				}
				read := analysis.Reads.Of("workflow_run_events")
				if read > pageSize+1+consolidationSlack {
					t.Fatalf("%s %s: the page after %d read %d events (budget %d):\n%s",
						history.runID, mode, after, read, pageSize+1+consolidationSlack, analysis.Plan)
				}
				largest = max(largest, read)
				rows, err := pool.Query(ctx, listRunEventsSQL, history.runID, after, pageSize, maxRunEventPageDataBytes)
				if err != nil {
					t.Fatal(err)
				}
				returned, lookAhead := 0, false
				for rows.Next() {
					var runID, eventID, schema, kind string
					var sequence int64
					var occurred time.Time
					var inPage bool
					var data []byte
					if err := rows.Scan(&runID, &sequence, &eventID, &schema, &kind, &occurred, &inPage, &data); err != nil {
						t.Fatal(err)
					}
					if inPage {
						delivered = append(delivered, sequence)
						returned++
					} else {
						lookAhead = true
					}
				}
				if err := rows.Err(); err != nil {
					t.Fatal(err)
				}
				if returned != 0 {
					after = delivered[len(delivered)-1]
				}
				more = lookAhead
			}
			want := make([]int64, 0, history.events)
			for sequence := int64(1); sequence <= history.events; sequence++ {
				want = append(want, sequence)
			}
			if !slices.Equal(delivered, want) || pages != int((history.events+pageSize-1)/pageSize) {
				t.Fatalf("%s %s: %d pages delivered %d events, want 1..%d", history.runID, mode, pages, len(delivered), history.events)
			}
			t.Logf("%s %s: %d pages, at most %d events read per page", history.runID, mode, pages, largest)
		}
	}

	termination := `{"outcome":"interrupted","code":"control_lease_expired","message":"allocation lost",` +
		`"retryable":true,"phase":"preparing","occurredAt":"2026-09-01T00:00:00Z"}`
	for _, read := range []struct {
		name string
		sql  string
		args []any
	}{
		// run-00550 is waiting with a preparing third Stage, run-00500
		// running with an aborting one, and run-00051 failed.
		{"active Stage of an active Run", getActiveStageExecutionSQL, []any{"run-00550"}},
		{"active Stage of a terminal Run", getActiveStageExecutionSQL, []any{"run-00051"}},
		{"terminate the active Stages", terminateActiveStagesSQL, []any{"run-00550", termination, "contractor/v1alpha1", "abort-consolidation"}},
		{"commit the aborting Stages", commitAbortingStagesSQL, []any{"run-00500"}},
		{"accept the finalizing candidates", acceptFinalizingCandidatesSQL, []any{"run-00500"}},
	} {
		for _, mode := range explaintest.Modes {
			analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, read.sql, read.args...)
			if err != nil {
				t.Fatalf("%s %s: %v", read.name, mode, err)
			}
			stages := analysis.Reads.Of("stage_executions")
			if stages > 3+consolidationSlack {
				t.Fatalf("%s %s: read %d Stages (budget %d):\n%s", read.name, mode, stages, 3+consolidationSlack, analysis.Plan)
			}
			t.Logf("%s %s: read %d Stages", read.name, mode, stages)
		}
	}
}

// TestPostgresRunEventIDsStayUniqueAcrossRuns pins the UNIQUE event_id
// constraint that V409-007 keeps beside the (run_id, sequence_number) primary
// key: an event of another Run cannot reuse an event ID.
func TestPostgresRunEventIDsStayUniqueAcrossRuns(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	insert := func(runID string) error {
		createTestRun(t, ctx, store, runID)
		_, err := pool.Exec(ctx, `
INSERT INTO workflow_run_events (run_id, sequence_number, event_id, event_schema_version, kind, data)
VALUES ($1, 1000, 'event-shared', $2, $3, $4::jsonb)`,
			runID, contracts.APIVersion, RunEventLifecycleChanged, `{"runId":"`+runID+`","resource":"run","state":"running"}`)
		return err
	}
	if err := insert("run-event-id-first"); err != nil {
		t.Fatal(err)
	}
	var failure *pgconn.PgError
	if err := insert("run-event-id-second"); persistencepostgres.Classify(err) != persistencepostgres.ClassDuplicate ||
		!errors.As(err, &failure) || failure.ConstraintName != "workflow_run_events_event_id_key" {
		t.Fatalf("event ID reused by another Run: %v", err)
	}
}
