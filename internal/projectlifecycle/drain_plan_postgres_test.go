package projectlifecycle

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The drain fixture is a database of several owners with a mostly terminal
// Run history. drainOwners owners have drainProjects Projects each; Runs go
// round the owners and over the first four Projects of each, and one Run in
// seven belongs to no Project. One Run in drainActiveEvery is active, as in
// a busy Queue, except in drainTarget, which has drained: all its Runs are
// terminal and released. Every Run has three Stages with one allocation
// each, and only the running Stage of a running Run still holds its
// allocation, so the unreleased allocations are the live ones. Audits belong
// to the other Projects, a few of them still running. drainEmpty has no Run
// and no Audit, like a Project in its Artifact purge. The history then grows
// to drainGrownRuns. Every table stays within the 30000 rows that ANALYZE
// samples whole at the default statistics target, so its statistics, and
// with them the plans, are the same on every run.
const (
	drainOwners      = 6
	drainProjects    = 5
	drainRuns        = 6000
	drainGrownRuns   = 9999
	drainActiveEvery = 50
	drainAudits      = 600
	drainTarget      = "project-0-0"
	drainEmpty       = "project-0-4"
	drainOwner       = "owner-0"
	// drainPlanningEntries bounds the index entries the planner reads to
	// find a column's current extremes (see explaintest); row budgets add it.
	drainPlanningEntries = 8
)

// The statements as they were before V407-025, kept to record their plans on
// the same fixture.
const (
	originalWaitForDrainSQL = `
SELECT EXISTS (
    SELECT 1 FROM workflow_runs
    WHERE project_id = $1
      AND state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling')
) OR EXISTS (
    SELECT 1
    FROM workflow_runs AS run
    JOIN stage_executions AS execution ON execution.run_id = run.run_id
    JOIN stage_allocations AS allocation
      ON allocation.stage_execution_id = execution.stage_execution_id
    WHERE run.project_id = $1
      AND allocation.release_completed_at IS NULL
) OR EXISTS (
    SELECT 1 FROM audits WHERE project_id = $1
)`
	originalPurgeRemainsSQL = `
SELECT EXISTS (SELECT 1 FROM workflow_runs WHERE project_id = $1),
       EXISTS (SELECT 1 FROM audits WHERE project_id = $1)`
)

// drainProjectsSQL creates the Projects: $1 owners with $2 Projects each.
const drainProjectsSQL = `
INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
SELECT 'project-' || owner || '-' || project, 'owner-' || owner, 'project', 'Drain fixture',
       'create-' || project, 'sha256:' || repeat('a', 64)
FROM generate_series(0, $1::integer - 1) AS owner, generate_series(0, $2::integer - 1) AS project`

// drainRunsSQL adds Runs $2 to $3 as the fixture describes; $1 is a valid
// RuntimeConfig snapshot, $4 the owner count, $5 the active interval, 0 for
// none, and $6 the Project that keeps no active Run.
const drainRunsSQL = `
INSERT INTO workflow_runs (
    run_id, owner_id, project_id, workflow_name, workflow_version, workflow_schema_version,
    workflow_snapshot, parameters, runtime_labels, runtime_config_snapshot, scheduler_lane_class,
    state, state_reason_code, created_at, updated_at, started_at, finished_at
)
SELECT 'run-' || lpad(i::text, 5, '0'), 'owner-' || (i % $4::integer), run.project_id,
       'drain', '1', 'contractor/v1alpha1', '{}'::jsonb, '{}'::jsonb, '{}'::text[], $1::jsonb, 'model',
       run.state, 'fixture', run.created_at, run.created_at,
       CASE WHEN run.state NOT IN ('initializing', 'pending') THEN run.created_at END,
       CASE WHEN run.state IN ('succeeded', 'failed') THEN run.created_at + interval '1 minute' END
FROM generate_series($2::integer, $3::integer) AS i
CROSS JOIN LATERAL (
    SELECT CASE WHEN i % 7 = 3 THEN NULL ELSE 'project-' || (i % $4::integer) || '-' || ((i / $4::integer) % 4) END
               AS project_id,
           timestamptz '2026-09-01 00:00:00+00' + i * interval '1 second' AS created_at
) AS placed
CROSS JOIN LATERAL (
    SELECT placed.project_id, placed.created_at,
           CASE
               WHEN $5::integer > 0 AND i % $5::integer = 0 AND placed.project_id IS DISTINCT FROM $6::text
                   THEN (ARRAY['initializing', 'pending', 'running', 'waiting'])[(i / $5::integer) % 4 + 1]
               WHEN i % 2 = 1 THEN 'failed'
               ELSE 'succeeded'
           END AS state
) AS run`

// drainStagesSQL gives each Run from $3 to $4 (by ID) three Stages: the third
// Stage of a running Run is preparing, every other one interrupted with
// StageTermination $1. $2 is a valid StageContext snapshot.
const drainStagesSQL = `
INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt,
    stage_spec_schema_version, stage_spec_snapshot, stage_context_schema_version, stage_context_snapshot,
    state, state_reason_code, termination_schema_version, stage_termination,
    abort_id, abort_deadline, terminal_at, created_at, updated_at
)
SELECT run.run_id || '-stage-' || stage, run.run_id, 'stage-' || stage, 1,
       'contractor/v1alpha1', '{}'::jsonb, 'contractor/v1alpha1', $2::jsonb,
       CASE WHEN live THEN 'preparing' ELSE 'interrupted' END,
       CASE WHEN live THEN 'created' ELSE 'termination_committed' END,
       CASE WHEN NOT live THEN 'contractor/v1alpha1' END,
       CASE WHEN NOT live THEN $1::jsonb END,
       CASE WHEN NOT live THEN 'abort-' || stage END,
       CASE WHEN NOT live THEN run.created_at END,
       CASE WHEN NOT live THEN run.created_at END,
       run.created_at, run.created_at
FROM workflow_runs AS run
CROSS JOIN generate_series(1, 3) AS stage
CROSS JOIN LATERAL (SELECT run.state = 'running' AND stage = 3 AS live) AS current
WHERE run.run_id BETWEEN $3 AND $4`

// drainAllocationsSQL gives each Stage of the Runs from $1 to $2 one
// allocation, released unless the Stage is still preparing.
const drainAllocationsSQL = `
INSERT INTO stage_allocations (
    allocation_id, stage_execution_id, logical_agent_name, namespace,
    agent_template_ref, worker_runtime_ref, runtime_agent_instance_id,
    release_attempted_at, release_completed_at
)
SELECT 'allocation-' || stage.stage_execution_id, stage.stage_execution_id, 'worker', 'worker',
       '{}'::jsonb, '{}'::jsonb, 'agent-instance',
       CASE WHEN stage.state <> 'preparing' THEN stage.created_at END,
       CASE WHEN stage.state <> 'preparing' THEN stage.created_at END
FROM stage_executions AS stage
WHERE stage.run_id BETWEEN $1 AND $2`

// drainAuditsSQL adds Audits $1 to $2 named audit-<n>, going round the
// Projects $3, all completed and released except one in fifty, which is
// running and holds its credentials.
const drainAuditsSQL = `
INSERT INTO audits (
    audit_id, owner_id, project_id, profile_name, profile_version, profile_digest,
    profile_snapshot, input_selection, baseline_snapshot, state, dispatch_state, hold_state,
    max_rounds, batch_size, max_items_per_round, max_items_total, max_submitted_runs,
    max_item_run_attempts, max_evidence_bytes, created_at, updated_at, started_at, finished_at, phase
)
SELECT 'audit-' || i, project.owner_id, project.project_id,
       'profile', '1', 'sha256:' || repeat('b', 64), '{}'::jsonb, '{}'::jsonb, '{}'::jsonb,
       CASE WHEN audit.running THEN 'active' ELSE 'completed' END,
       CASE WHEN audit.running THEN 'open' ELSE 'closed' END,
       CASE WHEN audit.running THEN 'held' ELSE 'released' END,
       1, 1, 1, 1, 1, 1, 1024, audit.created_at, audit.created_at, audit.created_at,
       CASE WHEN NOT audit.running THEN audit.created_at + interval '1 hour' END, 'rounds'
FROM generate_series($1::integer, $2::integer) AS i
JOIN projects AS project
  ON project.project_id = ($3::text[])[i % cardinality($3::text[]) + 1]
CROSS JOIN LATERAL (
    SELECT i % 50 = 0 AS running, timestamptz '2026-09-01 00:00:00+00' + i * interval '1 minute' AS created_at
) AS audit`

// populateDrainRuns adds Runs first to last, their Stages and allocations,
// without the per-row lifecycle events: no read under test looks at Run
// events, and appending one per row would dominate the load.
func populateDrainRuns(t *testing.T, ctx context.Context, pool *pgxpool.Pool, first, last, activeEvery int) {
	t.Helper()
	snapshot, err := json.Marshal(runtimeconfigtest.BuiltInRunSnapshot())
	if err != nil {
		t.Fatal(err)
	}
	firstID, lastID := drainRunID(first), drainRunID(last)
	termination := `{"outcome":"interrupted","code":"control_lease_expired","message":"allocation lost",` +
		`"retryable":true,"phase":"preparing","occurredAt":"2026-09-01T00:00:00Z"}`
	err = pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		for _, statement := range []struct {
			sql  string
			args []any
		}{
			// Deferred constraint triggers would otherwise block re-enabling.
			{sql: `SET CONSTRAINTS ALL IMMEDIATE`},
			{sql: `ALTER TABLE workflow_runs DISABLE TRIGGER workflow_runs_capture_lifecycle_event`},
			{sql: `ALTER TABLE stage_executions DISABLE TRIGGER stage_executions_capture_lifecycle_event`},
			{sql: drainRunsSQL, args: []any{snapshot, first, last, drainOwners, activeEvery, drainTarget}},
			{sql: drainStagesSQL, args: []any{termination, `{"parameters":{},"artifacts":{}}`, firstID, lastID}},
			{sql: drainAllocationsSQL, args: []any{firstID, lastID}},
			{sql: `ALTER TABLE workflow_runs ENABLE TRIGGER workflow_runs_capture_lifecycle_event`},
			{sql: `ALTER TABLE stage_executions ENABLE TRIGGER stage_executions_capture_lifecycle_event`},
		} {
			if _, err := tx.Exec(ctx, statement.sql, statement.args...); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("load Runs %d to %d: %v", first, last, err)
	}
}

func drainRunID(i int) string {
	id := []byte("run-00000")
	for position := len(id) - 1; i > 0; position, i = position-1, i/10 {
		id[position] = byte('0' + i%10)
	}
	return string(id)
}

// analyzeDrainFixture refreshes the statistics and removes the index entries
// of rolled-back changes. Without INDEX_CLEANUP ON, VACUUM skips the indexes
// when few pages hold dead rows, and the active-Run index the drain check
// reads would keep an entry for each Run a result case inserted.
func analyzeDrainFixture(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	if _, err := pool.Exec(ctx, `
VACUUM (INDEX_CLEANUP ON, ANALYZE) projects, workflow_runs, stage_executions, stage_allocations, audits`); err != nil {
		t.Fatal(err)
	}
}

// TestProjectDrainChecksReadOnlyTheProjectAndUnreleasedAllocations explains
// the drain check and the purge recheck of Project deletion on a realistic
// database, before and after its Run history grows by two thirds. The drain
// check of a drained Project reads, of workflow_runs, the index entries of
// the Project's Runs and of the active Runs, whose intersection finds no Run
// row, and at most one Run per unreleased allocation; of the Stage tables at
// most one row per unreleased allocation; and no Audit. The purge recheck of
// a Project without Runs descends one index of each table and reads no entry.
// Both answer as the statements before V407-025 did, which the test explains
// on the same data for the record. The result cases roll back changes that
// the second measurement must not count, so the fixture has a database of
// its own (explaintest.Database).
func TestProjectDrainChecksReadOnlyTheProjectAndUnreleasedAllocations(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	pool := explaintest.Database(t, ctx, databaseURL, func(ctx context.Context, pool *pgxpool.Pool) error {
		_, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{})
		return err
	})
	if _, err := pool.Exec(ctx, drainProjectsSQL, drainOwners, drainProjects); err != nil {
		t.Fatal(err)
	}
	populateDrainRuns(t, ctx, pool, 1, drainRuns, drainActiveEvery)
	var audited []string
	for owner := range drainOwners {
		for project := range drainProjects {
			if id := fmt.Sprintf("project-%d-%d", owner, project); id != drainTarget && id != drainEmpty {
				audited = append(audited, id)
			}
		}
	}
	if _, err := pool.Exec(ctx, drainAuditsSQL, 1, drainAudits, audited); err != nil {
		t.Fatal(err)
	}
	analyzeDrainFixture(t, ctx, pool)

	// The plans come first: the rolled-back changes of the result cases leave
	// dead index entries that only the next VACUUM removes.
	checkDrainPlans(t, ctx, pool, "fixture")
	checkDrainResults(t, ctx, pool)
	// More terminal history, in the target Project too, adds only the index
	// entries of the target's new Runs to what the checks read.
	populateDrainRuns(t, ctx, pool, drainRuns+1, drainGrownRuns, 0)
	analyzeDrainFixture(t, ctx, pool)
	checkDrainPlans(t, ctx, pool, "grown history")
}

// checkDrainResults compares both statements with their originals on the
// drained target, on a Project that is still busy and on the target after
// each change that blocks it, each in a transaction that is rolled back. A
// claim names its Project's owner; one naming another owner finds nothing.
func checkDrainResults(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	snapshot, err := json.Marshal(runtimeconfigtest.BuiltInRunSnapshot())
	if err != nil {
		t.Fatal(err)
	}
	// Run 6024 falls in the target Project; with an active interval of one
	// it is initializing.
	const activeTargetRun = 6024
	cases := []struct {
		name, projectID, ownerID string
		change                   func(pgx.Tx) (pgconn.CommandTag, error)
		blocked, runs, audits    bool
		foreignOwner             bool
	}{
		{name: "drained Project", projectID: drainTarget, ownerID: drainOwner, runs: true},
		{name: "Project without Runs", projectID: drainEmpty, ownerID: drainOwner},
		{name: "busy Project of another owner", projectID: "project-2-0", ownerID: "owner-2",
			blocked: true, runs: true, audits: true},
		{name: "another owner's claim", projectID: drainTarget, ownerID: "owner-1", foreignOwner: true},
		{name: "active Run", projectID: drainTarget, ownerID: drainOwner, blocked: true, runs: true,
			change: func(tx pgx.Tx) (pgconn.CommandTag, error) {
				return tx.Exec(ctx, drainRunsSQL, snapshot, activeTargetRun, activeTargetRun, drainOwners, 1, "")
			}},
		{name: "cancelling Run", projectID: drainTarget, ownerID: drainOwner, blocked: true, runs: true,
			change: func(tx pgx.Tx) (pgconn.CommandTag, error) {
				if _, err := tx.Exec(ctx, drainRunsSQL, snapshot, activeTargetRun, activeTargetRun, drainOwners, 1, ""); err != nil {
					return pgconn.CommandTag{}, err
				}
				return tx.Exec(ctx, `
UPDATE workflow_runs
   SET state = 'cancelling', cancellation_schema_version = 'contractor/v1alpha1',
       run_cancellation = '{"code":"user_cancelled","requestedAt":"2026-09-01T00:00:00Z"}', started_at = created_at
 WHERE run_id = $1`, drainRunID(activeTargetRun))
			}},
		{name: "unreleased allocation", projectID: drainTarget, ownerID: drainOwner, blocked: true, runs: true,
			change: func(tx pgx.Tx) (pgconn.CommandTag, error) {
				return tx.Exec(ctx, `
UPDATE stage_allocations SET release_completed_at = NULL
WHERE allocation_id = (SELECT min(allocation.allocation_id)
                         FROM stage_allocations AS allocation
                         JOIN stage_executions AS execution USING (stage_execution_id)
                         JOIN workflow_runs AS run USING (run_id)
                        WHERE run.project_id = $1)`, drainTarget)
			}},
		{name: "Audit", projectID: drainEmpty, ownerID: drainOwner, blocked: true, audits: true,
			change: func(tx pgx.Tx) (pgconn.CommandTag, error) {
				return tx.Exec(ctx, drainAuditsSQL, drainAudits+1, drainAudits+1, []string{drainEmpty})
			}},
		{name: "Audit being deleted", projectID: drainEmpty, ownerID: drainOwner, blocked: true, audits: true,
			change: func(tx pgx.Tx) (pgconn.CommandTag, error) {
				if _, err := tx.Exec(ctx, drainAuditsSQL, drainAudits+1, drainAudits+1, []string{drainEmpty}); err != nil {
					return pgconn.CommandTag{}, err
				}
				return tx.Exec(ctx, `
UPDATE audits SET state = 'deleting', deletion_requested_at = updated_at, dispatch_state = 'closed', hold_state = 'released'
WHERE project_id = $1`, drainEmpty)
			}},
	}
	for _, test := range cases {
		err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
			if test.change != nil {
				if tag, err := test.change(tx); err != nil || tag.RowsAffected() != 1 {
					t.Fatalf("%s: change = (%v, %v), want one row", test.name, tag, err)
				}
			}
			var blocked, originalBlocked, runs, audits, originalRuns, originalAudits bool
			for _, read := range []struct {
				sql  string
				args []any
				into []any
			}{
				{waitForDrainSQL, []any{test.projectID, test.ownerID}, []any{&blocked}},
				{originalWaitForDrainSQL, []any{test.projectID}, []any{&originalBlocked}},
				{purgeRemainsSQL, []any{test.projectID, test.ownerID}, []any{&runs, &audits}},
				{originalPurgeRemainsSQL, []any{test.projectID}, []any{&originalRuns, &originalAudits}},
			} {
				if err := tx.QueryRow(ctx, read.sql, read.args...).Scan(read.into...); err != nil {
					return err
				}
			}
			// The originals do not name the owner, so they answer the case
			// of another owner for the Project's own.
			if !test.foreignOwner &&
				(originalBlocked != test.blocked || originalRuns != test.runs || originalAudits != test.audits) {
				t.Errorf("%s: before V407-025 drain blocked = %t and purge recheck = (%t, %t), want %t and (%t, %t)",
					test.name, originalBlocked, originalRuns, originalAudits, test.blocked, test.runs, test.audits)
			}
			if blocked != test.blocked || runs != test.runs || audits != test.audits {
				t.Errorf("%s: drain blocked = %t and purge recheck = (%t, %t), want %t and (%t, %t)",
					test.name, blocked, runs, audits, test.blocked, test.runs, test.audits)
			}
			return errRollback
		})
		if !errors.Is(err, errRollback) {
			t.Fatalf("%s: %v", test.name, err)
		}
	}
}

var errRollback = errors.New("rolled back")

// checkDrainPlans explains both statements on the target and the empty
// Project in their custom and generic plans and holds them to their bounds.
// The row budgets count what explaintest.Reads counts: the rows sequential
// scans return and the entries index scans return, a bitmap's inputs
// included, so a row an inner scan reads in a few of its loops counts
// although EXPLAIN rounds its rows per loop to zero.
func checkDrainPlans(t *testing.T, ctx context.Context, pool *pgxpool.Pool, label string) {
	t.Helper()
	// projectRuns are the target's Runs, all terminal, and activeRuns every
	// owner's active Runs: the two inputs of the bitmap intersection that
	// finds the target's active Runs.
	var unreleased, projectRuns, activeRuns int64
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM stage_allocations WHERE release_completed_at IS NULL),
       (SELECT count(*) FROM workflow_runs WHERE owner_id = $2 AND project_id = $1),
       (SELECT count(*) FROM workflow_runs
         WHERE state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling'))`,
		drainTarget, drainOwner).Scan(&unreleased, &projectRuns, &activeRuns); err != nil {
		t.Fatal(err)
	}
	if unreleased == 0 || projectRuns == 0 || activeRuns == 0 {
		t.Fatalf("the fixture has %d unreleased allocations, %d Runs in the target and %d active Runs, want some of each",
			unreleased, projectRuns, activeRuns)
	}
	explain := func(sql string, args ...any) explaintest.Analysis {
		t.Helper()
		analysis, err := explaintest.Explain(ctx, pool, sql, args...)
		if err != nil {
			t.Fatalf("%s: explain: %v", label, err)
		}
		return analysis
	}
	reads := func(analysis explaintest.Analysis) string {
		var counts []string
		for _, table := range []string{"workflow_runs", "stage_executions", "stage_allocations", "audits"} {
			counts = append(counts, fmt.Sprintf("%s %d", table, analysis.Reads.Of(table)))
		}
		return strings.Join(counts, ", ")
	}

	drain, originalDrain := explain(waitForDrainSQL, drainTarget, drainOwner), explain(originalWaitForDrainSQL, drainTarget)
	t.Logf("%s, %d unreleased allocations, %d Runs in the target, %d active Runs: drain check of the drained Project, reads %s, %d blocks:\n%s\nbefore V407-025, reads %s, %d blocks:\n%s",
		label, unreleased, projectRuns, activeRuns, reads(drain), drain.Plan.Blocks(), drain.Plan,
		reads(originalDrain), originalDrain.Plan.Blocks(), originalDrain.Plan)
	for table, budget := range map[string]int64{
		// The index entries of the target's Runs and of the active Runs, and
		// one Run for each unreleased allocation.
		"workflow_runs": projectRuns + activeRuns + unreleased + drainPlanningEntries,
		// One Stage and one allocation for each unreleased allocation.
		"stage_executions":  unreleased + drainPlanningEntries,
		"stage_allocations": unreleased + drainPlanningEntries,
		// The target has no Audit, and no other Project's Audit is read.
		"audits": 0,
	} {
		if read := drain.Reads.Of(table); read > budget {
			t.Errorf("%s: the drain check read %d rows of %s, budget %d:\n%s", label, read, table, budget, drain.Plan)
		}
	}
	if drain.Plan.Blocks() >= originalDrain.Plan.Blocks() {
		t.Errorf("%s: the drain check reads %d blocks, no fewer than the %d it read before V407-025",
			label, drain.Plan.Blocks(), originalDrain.Plan.Blocks())
	}

	purge, originalPurge := explain(purgeRemainsSQL, drainEmpty, drainOwner), explain(originalPurgeRemainsSQL, drainEmpty)
	t.Logf("%s: purge recheck of the Project without Runs, reads %s, %d blocks:\n%s\nbefore V407-025, %d blocks:\n%s",
		label, reads(purge), purge.Plan.Blocks(), purge.Plan, originalPurge.Plan.Blocks(), originalPurge.Plan)
	// One descent of an index of at most three levels per table, which
	// finds no entry.
	if purge.Plan.Blocks() > 6 || purge.Reads.Of("workflow_runs") != 0 || purge.Reads.Of("audits") != 0 ||
		!purge.Plan.Uses("workflow_runs_owner_project_created_idx") {
		t.Errorf("%s: the purge recheck reads more than one index descent per table, reads %s:\n%s",
			label, reads(purge), purge.Plan)
	}

	conn, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Release()
	// The generic plan, which a cached statement may reuse for any Project,
	// is held only on the Run side: it may scan the few hundred Audits of
	// the fixture, whose check V407-025 did not change, when a vacuum cannot
	// mark their pages all-visible because another session's snapshot is
	// older.
	runTables := []string{"workflow_runs", "stage_executions", "stage_allocations"}
	for name, sql := range map[string]string{"drain check": waitForDrainSQL, "purge recheck": purgeRemainsSQL} {
		generic, err := explaintest.ExplainGeneric(ctx, conn.Conn().PgConn(), sql)
		if err != nil {
			t.Fatalf("%s: generic %s plan: %v", label, name, err)
		}
		t.Logf("%s: generic %s plan:\n%s", label, name, generic)
		if scans := generic.SeqScans(); !generic.Uses("workflow_runs_owner_project_created_idx") ||
			slices.ContainsFunc(scans, func(table string) bool { return slices.Contains(runTables, table) }) {
			t.Errorf("%s: the generic %s plan does not find the Project's Runs by owner and Project:\n%s", label, name, generic)
		}
	}
}
