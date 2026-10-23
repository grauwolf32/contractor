package postgres

import (
	"os"
	"testing"
)

// Migration 000030 marks the allocations that have a stored execution report
// as having received one, and from then on every inserted report marks its
// allocation (OPS-R183).
func TestPostgresAllocationReportReceivedMigrationMarksReportedAllocations(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 29)
	if _, err := pool.Exec(ctx, `INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version, workflow_schema_version, workflow_snapshot,
    parameters, runtime_labels, runtime_config_snapshot, scheduler_lane_class, state, state_reason_code,
    started_at
) VALUES ('run-reports', 'user', 'workflow', '1', 'v1', '{}', '{}', '{}', $1::jsonb, 'model', 'running', 'test',
          clock_timestamp())`, builtInRunRuntimeSnapshot); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt, stage_spec_schema_version, stage_spec_snapshot,
    stage_context_schema_version, stage_context_snapshot, state, state_reason_code
) VALUES ('stage-reports', 'run-reports', 'stage', 1, 'v1', '{}', 'v1', '{}', 'preparing', 'test')`); err != nil {
		t.Fatal(err)
	}
	report := func(allocation string) {
		t.Helper()
		if _, err := pool.Exec(ctx, `INSERT INTO allocation_execution_reports (
    report_id, stage_execution_id, allocation_id, logical_agent_name, report_schema_version, report
) VALUES ($1 || '-report', 'stage-reports', $1, $1, 'v1',
          jsonb_build_object('reportId', $1 || '-report', 'allocationId', $1, 'worker', '{}'::jsonb, 'runtime', '{}'::jsonb))`,
			allocation); err != nil {
			t.Fatal(err)
		}
	}
	for _, allocation := range []string{"allocation-reported", "allocation-silent"} {
		if _, err := pool.Exec(ctx, `INSERT INTO stage_allocations (
    allocation_id, stage_execution_id, logical_agent_name, namespace, agent_template_ref,
    worker_runtime_ref, runtime_agent_instance_id, performance_collection_policy
) VALUES ($1, 'stage-reports', $1, 'stage', '{}', '{}', 'instance', 'requested')`, allocation); err != nil {
			t.Fatal(err)
		}
	}
	report("allocation-reported")

	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	received := func(allocation string) bool {
		t.Helper()
		var value bool
		if err := pool.QueryRow(ctx, `SELECT execution_report_received FROM stage_allocations WHERE allocation_id = $1`,
			allocation).Scan(&value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	if !received("allocation-reported") || received("allocation-silent") {
		t.Fatalf("migrated received flags = reported %v, silent %v", received("allocation-reported"), received("allocation-silent"))
	}
	report("allocation-silent")
	if !received("allocation-silent") {
		t.Fatal("an inserted report did not mark its allocation")
	}
}
