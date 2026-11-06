package postgres

import (
	"os"
	"testing"
)

func TestGatewayRecoveryMemoryMigrationPreservesRuns(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, dsn)
	migrations, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	if len(migrations) < 14 || migrations[13].name != "000014_gateway_recovery_memory.sql" {
		t.Fatal("missing memory recovery migration")
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:6]); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, auditScopeMigrationFixtureSQL); err != nil {
		t.Fatal(err)
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:13]); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO workflow_runs(run_id,owner_id,workflow_name,workflow_version,workflow_schema_version,workflow_snapshot,parameters,runtime_labels,runtime_config_snapshot,scheduler_lane_class,state,state_reason_code)
VALUES ('recovery-run','owner','workflow','1','v1','{}','{}','{}',$1::jsonb,'model','initializing','test')`, builtInRunRuntimeSnapshot); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO gateway_recovery_routes(route_key,owner_id,blocked,failure_code) VALUES('route','owner',true,'model_unavailable')`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO gateway_run_routes(run_id,route_key) SELECT run_id,'route' FROM workflow_runs LIMIT 1`); err != nil {
		t.Fatal(err)
	}
	snapshot := func() string {
		t.Helper()
		var result string
		if err := pool.QueryRow(ctx, `SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id)::text FROM workflow_runs r`).Scan(&result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	before := snapshot()
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:14]); err != nil {
		t.Fatal(err)
	}
	if snapshot() != before {
		t.Fatal("migration changed Run business state")
	}
	for _, table := range []string{"gateway_recovery_routes", "gateway_run_routes", "gateway_allocation_routes", "gateway_recovery_waits", "gateway_recovery_failures"} {
		var absent bool
		if err := pool.QueryRow(ctx, `SELECT to_regclass($1) IS NULL`, table).Scan(&absent); err != nil || !absent {
			t.Fatalf("table %s remains: %v", table, err)
		}
	}
}
