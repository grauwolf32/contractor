package postgres

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"os"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresIntegrationMigrationsAndConstraints(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	first, second := isolatedPools(t, ctx, databaseURL)
	results := make([]MigrationResult, 2)
	errorsFound := make([]error, 2)
	var wait sync.WaitGroup
	for index, pool := range []*pgxpool.Pool{first, second} {
		wait.Add(1)
		go func(index int, pool *pgxpool.Pool) {
			defer wait.Done()
			results[index], errorsFound[index] = ApplyMigrations(ctx, pool)
		}(index, pool)
	}
	wait.Wait()
	for _, err := range errorsFound {
		if err != nil {
			t.Fatalf("concurrent ApplyMigrations: %v", err)
		}
	}
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	latest := available[len(available)-1].version
	applied := len(results[0].AppliedVersions) + len(results[1].AppliedVersions)
	if applied != len(available) || results[0].CurrentVersion != latest || results[1].CurrentVersion != latest {
		t.Fatalf("concurrent migration results = %+v, want one application of every version", results)
	}
	again, err := ApplyMigrations(ctx, first)
	if err != nil {
		t.Fatalf("second idempotent migration: %v", err)
	}
	if len(again.AppliedVersions) != 0 || again.CurrentVersion != latest {
		t.Fatalf("idempotent migration result = %+v", again)
	}

	tables := schemaTables(t, ctx, first)
	wantTables := []string{
		"allocation_execution_reports",
		"artifact_binding_revisions", "artifact_bindings", "artifact_blobs",
		"artifact_lineage", "artifact_pins", "artifact_scopes", "artifact_versions",
		"contractor_schema_migrations", "planner_events",
		"planner_execution_reports", "planner_sessions", "stage_allocations",
		"stage_execution_reports", "stage_executions", "stage_metrics",
		"stage_transition_decisions", "workflow_runs",
	}
	if strings.Join(tables, ",") != strings.Join(wantTables, ",") {
		t.Fatalf("schema tables = %v, want %v", tables, wantTables)
	}
	for _, name := range tables {
		if name == "runtime_agents" || strings.Contains(name, "heartbeat") {
			t.Fatalf("durable Runtime Agent liveness table exists: %q", name)
		}
	}

	_, err = first.Exec(ctx, `
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    state, state_reason_code
) VALUES ('invalid-state', 'user', 'workflow', '1', 'v1', '{}', '{}', 'bogus', 'test')`)
	assertSQLState(t, err, "23514")

	_, err = first.Exec(ctx, `
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    state, state_reason_code
) VALUES ('run-terminal-shape', 'user', 'workflow', '1', 'v1', '{}', '{}', 'succeeded', 'test')`)
	assertSQLState(t, err, "23514")

	_, err = first.Exec(ctx, `
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    state, state_reason_code
) VALUES ('run-for-stage', 'user', 'workflow', '1', 'v1', '{}', '{}', 'initializing', 'test')`)
	if err != nil {
		t.Fatalf("insert valid parent Run: %v", err)
	}
	_, err = first.Exec(ctx, `
INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt,
    stage_spec_schema_version, stage_spec_snapshot,
    stage_context_schema_version, stage_context_snapshot,
    state, state_reason_code, terminal_at
) VALUES (
    'invalid-terminal-stage', 'run-for-stage', 'build', 1,
    'v1', '{}', 'v1', '{}', 'succeeded', 'test', clock_timestamp()
)`)
	assertSQLState(t, err, "23514")
}

func isolatedPools(
	t *testing.T,
	ctx context.Context,
	databaseURL string,
) (*pgxpool.Pool, *pgxpool.Pool) {
	t.Helper()
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
	schema := "contractor_test_" + randomHex(t, 8)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatalf("create isolated schema: %v", err)
	}

	open := func() *pgxpool.Pool {
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
		return pool
	}
	first := open()
	second := open()
	t.Cleanup(func() {
		first.Close()
		second.Close()
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop isolated schema: %v", err)
		}
		admin.Close()
	})
	return first, second
}

func schemaTables(t *testing.T, ctx context.Context, pool *pgxpool.Pool) []string {
	t.Helper()
	rows, err := pool.Query(ctx, `
SELECT table_name
FROM information_schema.tables
WHERE table_schema = current_schema()
ORDER BY table_name`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var result []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			t.Fatal(err)
		}
		result = append(result, name)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	sort.Strings(result)
	return result
}

func assertSQLState(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Fatalf("SQL succeeded, want SQLSTATE %s", want)
	}
	if got := SQLState(err); got != want {
		t.Fatalf("SQLSTATE = %q, want %q (error: %v)", got, want, err)
	}
}

func randomHex(t *testing.T, bytes int) string {
	t.Helper()
	buffer := make([]byte, bytes)
	if _, err := rand.Read(buffer); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(buffer)
}
