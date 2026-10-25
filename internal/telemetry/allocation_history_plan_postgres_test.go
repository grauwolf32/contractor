package telemetry_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Keep the original statement as a semantic oracle and a negative read-budget
// control. Production statements are obtained through the Repository call below.
const originalAllocationHistorySQL = `WITH candidates AS (
    SELECT allocation.allocation_id, execution.run_id, execution.stage_execution_id,
           execution.stage_name, allocation.logical_agent_name, execution.state,
           execution.terminal_at, allocation.performance_collection_policy,
           allocation.release_completed_at, allocation.execution_report_received
    FROM workflow_runs AS run
    JOIN stage_executions AS execution ON execution.run_id = run.run_id
    JOIN stage_allocations AS allocation
      ON allocation.stage_execution_id = execution.stage_execution_id
    WHERE run.owner_id = $1
      AND execution.terminal_at IS NOT NULL
      AND (NOT $3::boolean OR run.run_id = $2)
      AND (NOT $6::boolean OR
           (execution.terminal_at, allocation.allocation_id) <= ($4::timestamptz, $5::text))
      AND (NOT $9::boolean OR
           (execution.terminal_at, allocation.allocation_id) < ($7::timestamptz, $8::text))
    ORDER BY execution.terminal_at DESC, allocation.allocation_id DESC
    LIMIT $10
)
SELECT candidate.allocation_id, candidate.run_id, candidate.stage_execution_id,
       candidate.stage_name, candidate.logical_agent_name, candidate.state,
       candidate.terminal_at, candidate.performance_collection_policy,
       candidate.release_completed_at, candidate.execution_report_received,
       effective.report_id IS NOT NULL,
       effective.reported_allocation_id, effective.resources
FROM candidates AS candidate
LEFT JOIN LATERAL (
    SELECT report.report_id,
           CASE WHEN jsonb_typeof(report.report->'allocationId') = 'string'
                THEN report.report->>'allocationId' END AS reported_allocation_id,
           report.report->'runtime'->'resources' AS resources
    FROM allocation_execution_reports AS report
    WHERE report.allocation_id = candidate.allocation_id
      AND report.expires_at > statement_timestamp()
    ORDER BY COALESCE(
                 (report.report->'worker'->>'complete')::boolean
                 AND (report.report->'runtime'->>'complete')::boolean,
                 false
             ) DESC,
             report.received_at DESC, report.report_id
    LIMIT 1
) AS effective ON true
ORDER BY candidate.terminal_at DESC, candidate.allocation_id DESC`

const (
	allocationHistoryRuns   = 6000
	allocationHistoryOwners = 6
	allocationHistoryGroup  = 3
	allocationHistoryPage   = 51
)

// This fixture has 6,000 Runs across six interleaved owners, three terminal
// Stages per timestamp, one allocation per Stage, and 12,000 reports. Its
// tables fit ANALYZE's default sample, so plan statistics are deterministic.
// It owns a database because report receipt triggers update allocations.
func TestPostgresAllocationHistoryPagesKeepBoundedReadsAndExactCursors(t *testing.T) {
	ctx := t.Context()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	pool := explaintest.Database(t, ctx, databaseURL, func(ctx context.Context, pool *pgxpool.Pool) error {
		_, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{})
		return err
	})
	populateAllocationHistory(t, ctx, pool)
	for table, want := range map[string]int{
		"workflow_runs":                allocationHistoryRuns,
		"stage_executions":             allocationHistoryRuns * allocationHistoryGroup,
		"stage_allocations":            allocationHistoryRuns * allocationHistoryGroup,
		"allocation_execution_reports": allocationHistoryRuns * 2,
	} {
		var got int
		if err := pool.QueryRow(ctx, "SELECT count(*) FROM "+pgx.Identifier{table}.Sanitize()).Scan(&got); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("fixture %s has %d rows, want %d", table, got, want)
		}
	}
	initial := telemetry.AllocationResourceHistoryParams{OwnerID: "owner-0", Limit: allocationHistoryPage}
	var deepAt, upperAt time.Time
	var deepID, upperID string
	if err := pool.QueryRow(ctx, `SELECT execution.terminal_at, allocation.allocation_id
FROM workflow_runs AS run JOIN stage_executions AS execution USING (run_id)
JOIN stage_allocations AS allocation USING (stage_execution_id)
WHERE run.owner_id = $1
ORDER BY execution.terminal_at DESC, allocation.allocation_id DESC OFFSET 1499 LIMIT 1`, initial.OwnerID).Scan(&deepAt, &deepID); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT execution.terminal_at, allocation.allocation_id
FROM workflow_runs AS run JOIN stage_executions AS execution USING (run_id)
JOIN stage_allocations AS allocation USING (stage_execution_id)
WHERE run.owner_id = $1
ORDER BY execution.terminal_at DESC, allocation.allocation_id DESC LIMIT 1`, initial.OwnerID).Scan(&upperAt, &upperID); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		params telemetry.AllocationResourceHistoryParams
	}{
		{"first", initial},
		{"deep-cursor", telemetry.AllocationResourceHistoryParams{OwnerID: initial.OwnerID, Limit: allocationHistoryPage, AfterFinishedAt: &deepAt, AfterAllocationID: deepID}},
		{"deep-upper", telemetry.AllocationResourceHistoryParams{OwnerID: initial.OwnerID, Limit: allocationHistoryPage, UpperFinishedAt: &deepAt, UpperAllocationID: deepID}},
		{"deep-pinned", telemetry.AllocationResourceHistoryParams{OwnerID: initial.OwnerID, Limit: allocationHistoryPage, UpperFinishedAt: &upperAt, UpperAllocationID: upperID, AfterFinishedAt: &deepAt, AfterAllocationID: deepID}},
	}
	for _, mode := range explaintest.Modes {
		t.Run(string(mode), func(t *testing.T) {
			for _, test := range tests {
				t.Run(test.name, func(t *testing.T) { checkAllocationHistoryReads(t, ctx, pool, mode, test.params, false) })
			}
			for _, runID := range []string{"run-00006", "run-00001"} {
				for _, test := range tests {
					t.Run(runID+"/"+test.name, func(t *testing.T) {
						params := test.params
						params.RunID = runID
						checkAllocationHistoryReads(t, ctx, pool, mode, params, true)
					})
				}
			}
		})
	}
	t.Run("original-regression-controls", func(t *testing.T) {
		for _, test := range []struct {
			name   string
			mode   explaintest.Mode
			params telemetry.AllocationResourceHistoryParams
			budget int64
		}{
			{"deep-custom", explaintest.Custom, tests[1].params, allocationHistoryStageBudget(allocationHistoryPage)},
			{"run-generic", explaintest.Generic, telemetry.AllocationResourceHistoryParams{OwnerID: initial.OwnerID, RunID: "run-00006", Limit: allocationHistoryPage}, allocationHistoryGroup + 8},
		} {
			t.Run(test.name, func(t *testing.T) {
				plan, err := explaintest.ExplainPrepared(ctx, pool, test.mode, originalAllocationHistorySQL, originalAllocationHistoryArgs(test.params)...)
				if err != nil {
					t.Fatal(err)
				}
				reads := plan.Reads.Of("stage_executions")
				t.Logf("original Stage reads=%d, budget=%d", reads, test.budget)
				if reads <= test.budget {
					t.Fatalf("negative control no longer demonstrates regression: %d <= %d", reads, test.budget)
				}
			})
		}
	})
	t.Run("walk-parity", func(t *testing.T) {
		// Limit two deliberately splits the three-allocation timestamp groups.
		// Include an upper boundary inside a tie and an unauthorized exact Run.
		tiedUpperID := "allocation-run-03000-stage-2"
		tiedUpperAt := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC).Add(3000 * time.Second)
		for _, runID := range []string{"", "run-03000", "run-00001"} {
			for _, bounded := range []bool{false, true} {
				t.Run(fmt.Sprintf("run=%s/upper=%t", runID, bounded), func(t *testing.T) {
					params := telemetry.AllocationResourceHistoryParams{OwnerID: initial.OwnerID, RunID: runID, Limit: 2}
					if bounded {
						params.UpperFinishedAt = &tiedUpperAt
						params.UpperAllocationID = tiedUpperID
					}
					seen := map[string]bool{}
					count := 0
					for {
						got, err := telemetry.NewRepository(pool).ListAllocationResourceHistory(ctx, params)
						if err != nil {
							t.Fatal(err)
						}
						oracle := &allocationHistoryOriginalDB{DBTX: pool, params: params}
						want, err := telemetry.NewRepository(oracle).ListAllocationResourceHistory(ctx, params)
						if err != nil {
							t.Fatal(err)
						}
						if !reflect.DeepEqual(got, want) {
							t.Fatalf("page %d differs: got %+v, want %+v", count, got, want)
						}
						if len(got) == 0 {
							break
						}
						for _, item := range got {
							if seen[item.AllocationID] {
								t.Fatalf("repeated %s", item.AllocationID)
							}
							seen[item.AllocationID] = true
							count++
						}
						last := got[len(got)-1]
						params.AfterFinishedAt = &last.FinishedAt
						params.AfterAllocationID = last.AllocationID
						if count > allocationHistoryRuns*allocationHistoryGroup {
							t.Fatal("cursor did not terminate")
						}
					}
					wantCount := allocationHistoryRuns * allocationHistoryGroup / allocationHistoryOwners
					if bounded {
						wantCount = 1499
					}
					if runID == "run-03000" {
						wantCount = 3
						if bounded {
							wantCount = 2
						}
					}
					if runID == "run-00001" {
						wantCount = 0
					}
					if count != wantCount {
						t.Fatalf("walk returned %d rows, want %d", count, wantCount)
					}
				})
			}
		}
	})
}

// Allow the requested page, two terminal-time groups consumed by incremental
// sorting, six interleaved owners, and eight planner/index-boundary probes.
func allocationHistoryStageBudget(limit int) int64 {
	return int64((limit+2*allocationHistoryGroup)*allocationHistoryOwners + 8)
}

func checkAllocationHistoryReads(t *testing.T, ctx context.Context, pool *pgxpool.Pool, mode explaintest.Mode, params telemetry.AllocationResourceHistoryParams, filtered bool) {
	t.Helper()
	probe := &allocationHistoryPlanDB{Pool: pool, mode: mode}
	got, err := telemetry.NewRepository(probe).ListAllocationResourceHistory(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if probe.queries != 1 {
		t.Fatalf("measured %d statements, want one", probe.queries)
	}
	if len(got) > params.Limit {
		t.Fatalf("page exceeded limit: %d", len(got))
	}
	stageBudget := allocationHistoryStageBudget(params.Limit)
	runBudget := stageBudget
	allocationBudget := int64(params.Limit + 2*allocationHistoryGroup + 8)
	if filtered {
		stageBudget = allocationHistoryGroup + 8
		runBudget = 8
		allocationBudget = allocationHistoryGroup + 8
	}
	for table, budget := range map[string]int64{"stage_executions": stageBudget, "workflow_runs": runBudget, "stage_allocations": allocationBudget} {
		reads := probe.analysis.Reads.Of(table)
		t.Logf("%s reads=%d budget=%d", table, reads, budget)
		if reads > budget {
			t.Errorf("%s reads=%d exceed page-derived budget %d; plan=%+v", table, reads, budget, probe.analysis.Plan)
		}
	}
}

type allocationHistoryPlanDB struct {
	*pgxpool.Pool
	mode     explaintest.Mode
	analysis explaintest.Analysis
	queries  int
}

func (p *allocationHistoryPlanDB) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	p.queries++
	var err error
	p.analysis, err = explaintest.ExplainPrepared(ctx, p.Pool, p.mode, sql, args...)
	if err != nil {
		return nil, err
	}
	return p.Pool.Query(ctx, sql, args...)
}

type allocationHistoryOriginalDB struct {
	persistencepostgres.DBTX
	params telemetry.AllocationResourceHistoryParams
}

func (p *allocationHistoryOriginalDB) Query(ctx context.Context, _ string, _ ...any) (pgx.Rows, error) {
	return p.DBTX.Query(ctx, originalAllocationHistorySQL, originalAllocationHistoryArgs(p.params)...)
}
func originalAllocationHistoryArgs(params telemetry.AllocationResourceHistoryParams) []any {
	upperAt, afterAt := time.Time{}, time.Time{}
	if params.UpperFinishedAt != nil {
		upperAt = params.UpperFinishedAt.UTC()
	}
	if params.AfterFinishedAt != nil {
		afterAt = params.AfterFinishedAt.UTC()
	}
	return []any{params.OwnerID, params.RunID, params.RunID != "", upperAt, params.UpperAllocationID, params.UpperFinishedAt != nil, afterAt, params.AfterAllocationID, params.AfterFinishedAt != nil, params.Limit}
}

func populateAllocationHistory(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	snapshot, err := json.Marshal(runtimeconfigtest.BuiltInRunSnapshot())
	if err != nil {
		t.Fatal(err)
	}
	termination := `{"outcome":"interrupted","code":"control_lease_expired","message":"allocation lost","retryable":true,"phase":"preparing","occurredAt":"2026-09-01T00:00:00Z"}`
	report, err := json.Marshal(allocationEnvelope("fixture", "fixture", time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC), true).Report)
	if err != nil {
		t.Fatal(err)
	}
	err = pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		for _, step := range []struct {
			sql  string
			args []any
		}{
			{sql: `SET CONSTRAINTS ALL IMMEDIATE`},
			{sql: `ALTER TABLE workflow_runs DISABLE TRIGGER workflow_runs_capture_lifecycle_event`},
			{sql: `ALTER TABLE stage_executions DISABLE TRIGGER stage_executions_capture_lifecycle_event`},
			{sql: `INSERT INTO workflow_runs (
 run_id,owner_id,workflow_name,workflow_version,workflow_schema_version,workflow_snapshot,
 runtime_labels,runtime_config_snapshot,scheduler_lane_class,state,state_reason_code,finished_at)
SELECT 'run-'||lpad(i::text,5,'0'),'owner-'||(i % $2::integer),'history','1','contractor/v1alpha1','{}',
 '{}'::text[],$1::jsonb,'model','succeeded','fixture',timestamptz '2026-09-01 00:00:00+00' + i*interval '1 second'
FROM generate_series(1,$3::integer) AS i`, args: []any{snapshot, allocationHistoryOwners, allocationHistoryRuns}},
			{sql: `INSERT INTO stage_executions (
 stage_execution_id,run_id,stage_name,attempt,stage_spec_schema_version,stage_spec_snapshot,
 stage_context_schema_version,stage_context_snapshot,state,state_reason_code,
 termination_schema_version,stage_termination,abort_id,abort_deadline,terminal_at)
SELECT run_id||'-stage-'||stage,run_id,'stage-'||stage,1,'contractor/v1alpha1','{}',
 'contractor/v1alpha1','{"parameters":{},"artifacts":{}}','interrupted','termination_committed',
 'contractor/v1alpha1',$1::jsonb,'abort-'||stage,finished_at,finished_at
FROM workflow_runs CROSS JOIN generate_series(1,$2::integer) AS stage`, args: []any{termination, allocationHistoryGroup}},
			{sql: `INSERT INTO stage_allocations (
 allocation_id,stage_execution_id,logical_agent_name,namespace,agent_template_ref,worker_runtime_ref,
 runtime_agent_instance_id,performance_collection_policy,release_attempted_at,release_completed_at)
SELECT 'allocation-'||stage_execution_id,stage_execution_id,'builder','builder','{}','{}','fixture',
 'requested',terminal_at,terminal_at FROM stage_executions`},
			{sql: `INSERT INTO allocation_execution_reports (
 report_id,stage_execution_id,allocation_id,logical_agent_name,report_schema_version,report,received_at,expires_at)
SELECT allocation_id||'-report-'||complete,stage_execution_id,allocation_id,'builder','contractor/v1alpha1',
 jsonb_set(jsonb_set($1::jsonb||jsonb_build_object('reportId',allocation_id||'-report-'||complete,
 'allocationId',allocation_id),'{worker,complete}',to_jsonb(complete)),
 '{runtime,complete}',to_jsonb(complete)),
 statement_timestamp()-CASE WHEN complete THEN interval '2 minutes' ELSE interval '1 minute' END,
 statement_timestamp()+interval '1 day'
FROM stage_allocations CROSS JOIN (VALUES(true),(false)) AS versions(complete)
WHERE stage_execution_id LIKE '%-stage-1'`, args: []any{report}},
			{sql: `ALTER TABLE workflow_runs ENABLE TRIGGER workflow_runs_capture_lifecycle_event`},
			{sql: `ALTER TABLE stage_executions ENABLE TRIGGER stage_executions_capture_lifecycle_event`},
		} {
			if _, err := tx.Exec(ctx, step.sql, step.args...); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `VACUUM ANALYZE workflow_runs,stage_executions,stage_allocations,allocation_execution_reports`); err != nil {
		t.Fatal(err)
	}
}
