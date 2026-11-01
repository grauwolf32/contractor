package postgres_test

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/persistence/migrations"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestProjectFindingCutoverRejectsRetainedAuditRunsBeforeArchive(t *testing.T) {
	for _, outcome := range []string{"succeeded", "failed", "cancelled", "active", "unreleased", "uncollected"} {
		t.Run(outcome, func(t *testing.T) {
			ctx := t.Context()
			pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{})
			seedRetirementSchema(t, ctx, pool)
			tx, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer tx.Rollback(ctx)
			runState, executionState, disposition := outcome, "collected", "missing-output"
			switch outcome {
			case "failed":
				disposition = "execution-failed"
			case "cancelled":
				disposition = "execution-cancelled"
			case "active":
				runState = "running"
			case "unreleased":
				runState = "failed"
			case "uncollected":
				runState = "failed"
				executionState = "collecting"
			}
			terminal := "succeeded"
			if runState == "failed" || runState == "cancelled" {
				terminal = runState
			}
			_, err = tx.Exec(ctx, `INSERT INTO audit_executions(execution_id,audit_id,role,role_attempt,workflow_role,lane_class,manifest_ref,manifest_digest,submission_key,request_digest)
 VALUES('cutover-execution','migration-audit','discovery',1,'discovery','tool','{}','sha256:'||repeat('a',64),'cutover-submission','sha256:'||repeat('a',64))`)
			if err != nil {
				t.Fatal(err)
			}

			snapshot, err := json.Marshal(runtimeconfigtest.BuiltInRunSnapshot())
			if err != nil {
				t.Fatal(err)
			}
			for _, run := range []struct {
				id, state, mode string
				execution, key  *string
			}{
				{"cutover-run", runState, "audit-managed", ptr("cutover-execution"), ptr("cutover-submission")},
				{"ordinary-run", "succeeded", "ordinary", nil, nil},
			} {
				_, err = tx.Exec(ctx, `INSERT INTO workflow_runs(run_id,owner_id,project_id,workflow_name,workflow_version,workflow_schema_version,workflow_snapshot,runtime_labels,runtime_config_snapshot,scheduler_lane_class,state,state_reason_code,publication_mode,audit_execution_id,audit_submission_key,finished_at,cancellation_schema_version,run_cancellation)
 VALUES($1,'migration-owner','migration-project','fixture','1','v1','{}','{}',$2::jsonb,'tool',$3,'fixture',$4,$5,$6,CASE WHEN $3 IN ('succeeded','failed','cancelled') THEN clock_timestamp() END,CASE WHEN $3='cancelled' THEN 'v1' END,CASE WHEN $3='cancelled' THEN '{"code":"user_cancelled","requestedAt":"2026-11-02T00:00:00Z","requestedBy":"migration-owner"}'::jsonb END)`, run.id, snapshot, run.state, run.mode, run.execution, run.key)
				if err != nil {
					t.Fatal(err)
				}
			}
			_, err = tx.Exec(ctx, `UPDATE audit_executions SET run_id='cutover-run',state=$1,terminal_outcome=$2,terminal_run_generation='generation',terminal_run_sequence=1,terminal_observed_at=clock_timestamp(),run_provenance='{"schema":"contractor.audit.run-provenance.v1","runId":"cutover-run"}',
 collection_receipt_id=CASE WHEN $1='collected' THEN 'cutover-collection' END,
 collection_disposition=CASE WHEN $1='collected' THEN $3::text END,
 collection_retained_refs=CASE WHEN $1='collected' THEN '[]'::jsonb END,
 collection_request_digest=CASE WHEN $1='collected' THEN 'sha256:'||repeat('a',64) END,
 collected_at=CASE WHEN $1='collected' THEN clock_timestamp() END
 WHERE execution_id='cutover-execution'`, executionState, terminal, disposition)
			if err != nil {
				t.Fatal(err)
			}
			if outcome == "unreleased" {
				_, err = tx.Exec(ctx, `INSERT INTO stage_executions(stage_execution_id,run_id,stage_name,attempt,stage_spec_schema_version,stage_spec_snapshot,stage_context_schema_version,stage_context_snapshot,state,state_reason_code)
 VALUES('cutover-stage','cutover-run','fixture',1,'v1','{}','v1','{}','preparing','fixture');
 INSERT INTO stage_allocations(allocation_id,stage_execution_id,logical_agent_name,namespace,agent_template_ref,worker_runtime_ref,runtime_agent_instance_id)
 VALUES('cutover-allocation','cutover-stage','worker','worker','{}','{}','runtime')`)
				if err != nil {
					t.Fatal(err)
				}
			}
			if err = tx.Commit(ctx); err != nil {
				t.Fatal(err)
			}
			_, err = pg.ApplyMigrations(ctx, pool, pg.MigrationBudgets{})
			if pg.SQLState(err) != pg.SQLStateCheckViolation || !strings.Contains(err.Error(), "before switching") {
				t.Fatalf("initial cutover fence: %v", err)
			}
			var intact bool
			if err = pool.QueryRow(ctx, `SELECT to_regclass('project_findings') IS NULL AND (SELECT count(*) FROM workflow_runs)=2 AND (SELECT count(*) FROM audits)=1 AND NOT EXISTS(SELECT 1 FROM artifact_bindings WHERE namespace='historical-audits')`).Scan(&intact); err != nil || !intact {
				t.Fatalf("cutover was not atomic: %v %v", intact, err)
			}
			if outcome == "active" || outcome == "unreleased" || outcome == "uncollected" {
				err = rundeletion.New(pool).DeleteReleasedTerminalRun(ctx, "migration-owner", "cutover-run")
				var blocked *rundeletion.NotDeletableError
				if !errors.As(err, &blocked) {
					t.Fatalf("unsafe retirement: %v", err)
				}
				var count int
				if err = pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs`).Scan(&count); err != nil || count != 2 {
					t.Fatalf("unsafe purge changed Runs: %d %v", count, err)
				}
			}
			// The positive retirement is performed by the matching pre-cutover
			// Server during the disposable-restore rehearsal. Current code has
			// no old-schema reader. The archive test verifies the post-retirement
			// mixed ordinary-Run/Project boundary and migration replay.

		})
	}
}
func ptr(s string) *string { return &s }
func seedRetirementSchema(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	if _, err := pool.Exec(ctx, `CREATE TABLE contractor_schema_migrations(version bigint PRIMARY KEY,name text NOT NULL,checksum bytea NOT NULL,applied_at timestamptz NOT NULL DEFAULT clock_timestamp())`); err != nil {
		t.Fatal(err)
	}
	for i, name := range []string{"000001_baseline.sql", "000002_remove_scan_sweeps.sql"} {
		body, err := migrations.Files.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = pool.Exec(ctx, string(body)); err != nil {
			t.Fatal(err)
		}
		checksum := sha256.Sum256(body)
		if _, err = pool.Exec(ctx, `INSERT INTO contractor_schema_migrations(version,name,checksum) VALUES($1,$2,$3)`, i+1, name, checksum[:]); err != nil {
			t.Fatal(err)
		}
	}
	fixture, err := os.ReadFile("testdata/project-findings-migration.sql")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = pool.Exec(ctx, string(fixture)); err != nil {
		t.Fatal(err)
	}
}

func TestProjectFindingCutoverFlushesCollectedItemBackreferences(t *testing.T) {
	ctx := t.Context()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{})
	seedRetirementSchema(t, ctx, pool)
	fixture, err := os.ReadFile("testdata/project-finding-collected-items.sql")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = pool.Exec(ctx, string(fixture)); err != nil {
		t.Fatal(err)
	}
	if _, err = pg.ApplyMigrations(ctx, pool, pg.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	var history int
	if err = pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE namespace='historical-audits' AND name LIKE '%.execution_items.%'`).Scan(&history); err != nil || history != 1 {
		t.Fatalf("collected item history: %d %v", history, err)
	}
}
