package postgres

import (
	"os"
	"testing"
)

func TestProjectFindingMigrationArchivesAuditsAndDropsOldFindings(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, dsn)
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	if len(available) < 3 || available[2].name != "000003_project_findings.sql" {
		t.Fatal("expected Project Finding migration after the current baseline")
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:2]); err != nil {
		t.Fatal(err)
	}
	fixture, err := os.ReadFile("testdata/project-findings-migration.sql")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, string(fixture)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO workflow_runs(run_id,owner_id,project_id,workflow_name,workflow_version,workflow_schema_version,workflow_snapshot,runtime_labels,runtime_config_snapshot,scheduler_lane_class,state,state_reason_code,finished_at)
 VALUES('ordinary-retained-run','migration-owner','migration-project','fixture','1','v1','{}','{}',$1::jsonb,'tool','succeeded','fixture',clock_timestamp())`, builtInRunRuntimeSnapshot); err != nil {
		t.Fatal(err)
	}
	var ordinaryBefore string
	if err := pool.QueryRow(ctx, `SELECT to_jsonb(run)::text FROM workflow_runs run WHERE run_id='ordinary-retained-run'`).Scan(&ordinaryBefore); err != nil {
		t.Fatal(err)
	}
	var before string
	if err := pool.QueryRow(ctx, "SELECT jsonb_agg(to_jsonb(hold) ORDER BY receipt_id)::text FROM finding_proposal_audit_holds AS hold").Scan(&before); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	for _, check := range []struct {
		query string
		want  int
	}{
		{"SELECT count(*) FROM project_findings", 0},
		{"SELECT count(*) FROM audits", 0},
		{"SELECT count(*) FROM information_schema.tables WHERE table_schema=current_schema() AND table_name IN ('audit_findings','audit_finding_assessments','audit_finding_contributions')", 0},
		{"SELECT count(*) FROM artifact_bindings WHERE namespace='historical-audits' AND name LIKE '%.snapshot.%'", 1},
		{"SELECT count(*) FROM project_finding_sources", 0},
		{"SELECT count(*) FROM finding_proposal_receipts", 0},
		{"SELECT count(*) FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='finding_proposal_receipts' AND column_name='checks_projection' AND is_nullable='NO'", 1},
		{"SELECT count(*) FROM pg_indexes WHERE schemaname=current_schema() AND indexname='finding_checks_projection_pending_idx'", 0},
		{"SELECT count(*) FROM artifact_bindings WHERE namespace='historical-audits' AND name LIKE '%.review_decisions.%'", 2},
		{"SELECT count(*) FROM artifact_bindings WHERE namespace='historical-audits' AND name LIKE '%.finding_assessments.%'", 1},
		{"SELECT count(*) FROM artifact_blobs WHERE payload='frozen report snapshot'::bytea", 1},
	} {
		var got int
		if err := pool.QueryRow(ctx, check.query).Scan(&got); err != nil || got != check.want {
			t.Fatalf("%s: got %d want %d err=%v", check.query, got, check.want, err)
		}
	}
	var after string
	if err := pool.QueryRow(ctx, "SELECT jsonb_agg(fact->'original' ORDER BY fact#>>'{original,receipt_id}')::text FROM (SELECT convert_from(b.payload,'UTF8')::jsonb AS fact FROM artifact_binding_revisions r JOIN artifact_versions v USING(version_id) JOIN artifact_blobs b ON b.sha256=v.blob_sha256 WHERE r.namespace='historical-audits' AND r.name LIKE '%.finding_proposal_holds.%') AS history").Scan(&after); err != nil || before != after {
		t.Fatalf("historical exact Audit receipt selections changed: %v", err)
	}
	var confirmed, duplicate bool
	if err := pool.QueryRow(ctx, "SELECT bool_or(convert_from(b.payload,'UTF8')::jsonb#>>'{original,verdict}'='true_positive'),bool_or(convert_from(b.payload,'UTF8')::jsonb#>>'{original,verdict}'='duplicate') FROM artifact_binding_revisions r JOIN artifact_versions v USING(version_id) JOIN artifact_blobs b ON b.sha256=v.blob_sha256 WHERE r.namespace='historical-audits'").Scan(&confirmed, &duplicate); err != nil || !confirmed || !duplicate {
		t.Fatalf("original annotations lost: confirmed=%v duplicate=%v err=%v", confirmed, duplicate, err)
	}
	var ordinaryAfter string
	if err := pool.QueryRow(ctx, `SELECT to_jsonb(run)::text FROM workflow_runs run WHERE run_id='ordinary-retained-run'`).Scan(&ordinaryAfter); err != nil || ordinaryBefore != ordinaryAfter {
		t.Fatalf("ordinary Run changed: %v", err)
	}
	if result, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil || len(result.AppliedVersions) != 0 {
		t.Fatalf("replay: %+v %v", result, err)
	}

}

func TestProjectFindingMigrationRejectsUndrainedRunsAtomically(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, dsn)
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:2]); err != nil {
		t.Fatal(err)
	}
	fixture, err := os.ReadFile("testdata/project-findings-migration.sql")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, string(fixture)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO workflow_runs(run_id,owner_id,project_id,workflow_name,workflow_version,workflow_schema_version,workflow_snapshot,runtime_labels,runtime_config_snapshot,scheduler_lane_class,state,state_reason_code)
 VALUES('undrained-run','migration-owner','migration-project','test','1','v1','{}','{}',$1::jsonb,'tool','pending','fixture')`, builtInRunRuntimeSnapshot); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); SQLState(err) != SQLStateCheckViolation {
		t.Fatalf("undrained cutover: %v", err)
	}
	var intact bool
	if err := pool.QueryRow(ctx, `SELECT to_regclass('project_findings') IS NULL AND to_regclass('audit_findings') IS NOT NULL
 AND (SELECT count(*) FROM audits)=1 AND (SELECT count(*) FROM finding_proposal_receipts)=2
 AND NOT EXISTS(SELECT 1 FROM artifact_bindings WHERE namespace='historical-audits')`).Scan(&intact); err != nil || !intact {
		t.Fatalf("failed cutover left partial schema/archive: intact=%v err=%v", intact, err)
	}
	// Finishing the same Run makes the original migration retryable without a
	// compatibility reader or any repair writes to historical Audit data.
	if _, err := pool.Exec(ctx, `UPDATE workflow_runs SET state='failed',state_reason_code='fixture_finished',finished_at=clock_timestamp() WHERE run_id='undrained-run'`); err != nil {
		t.Fatal(err)
	}

	if _, err := pool.Exec(ctx, `ALTER TABLE audit_executions DISABLE TRIGGER USER;
 INSERT INTO audit_executions(execution_id,audit_id,role,role_attempt,workflow_role,lane_class,manifest_ref,manifest_digest,submission_key,request_digest)
 VALUES('uncollected-execution','migration-audit','discovery',1,'discovery','model','{}','sha256:'||repeat('a',64),'uncollected','sha256:'||repeat('a',64));
 ALTER TABLE audit_executions ENABLE TRIGGER USER;`); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); SQLState(err) != SQLStateCheckViolation {
		t.Fatalf("uncollected cutover: %v", err)
	}
	if err := pool.QueryRow(ctx, `SELECT to_regclass('project_findings') IS NULL AND (SELECT count(*) FROM audits)=1 AND NOT EXISTS(SELECT 1 FROM artifact_bindings WHERE namespace='historical-audits')`).Scan(&intact); err != nil || !intact {
		t.Fatalf("uncollected cutover was not atomic: intact=%v err=%v", intact, err)
	}
	// The unsubmitted intent is cancelled before retrying the one-time switch.
	if _, err := pool.Exec(ctx, `DELETE FROM audit_executions WHERE execution_id='uncollected-execution'`); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
}
