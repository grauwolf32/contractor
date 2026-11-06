package postgres

import (
	"os"
	"strings"
	"testing"
)

func TestPostgresAuditPipelineRetirementPreservesCurrentWorkAndExactHistory(t *testing.T) {
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
	const previous = 11
	if len(migrations) <= previous || migrations[previous].name != "000012_remove_audit_pipeline.sql" {
		t.Fatal("missing retirement migration")
	}
	if _, err = applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:previous]); err != nil {
		t.Fatal(err)
	}
	if _, err = pool.Exec(ctx, auditScopeMigrationFixtureSQL); err != nil {
		t.Fatal(err)
	}
	for _, state := range []string{"active", "paused", "completed"} {
		id := "current-" + state
		if _, err = pool.Exec(ctx, `INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,profile_snapshot,input_selection,baseline_snapshot,state,started_at,finished_at,max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes)
VALUES($1,'owner','scope-project','common','1','sha256:'||repeat('b',64),'{}','{}','{}',$2,clock_timestamp(),CASE WHEN $2='completed' THEN clock_timestamp() END,1,1,1,1,2,1,1048576)`, id, state); err != nil {
			t.Fatal(err)
		}
		if _, err = pool.Exec(ctx, `INSERT INTO audit_tasks(task_id,audit_id,source_key,source,document,batch_digest,ordinal,iteration,state,result,gap_code)
VALUES($1,$1,'sha256:'||repeat('c',64),'{"kind":"initial","key":"work"}',json_build_object('schema','contractor.audit.task.v1','task_id',$1::text,'audit_id',$1::text,'workflow','inspect','objective','Retain original observation','inputs','{}'::json,'parameters','{}'::json),'sha256:'||repeat('d',64),1,1,CASE WHEN $2='completed' THEN 'settled' ELSE 'ready' END,NULL,CASE WHEN $2='completed' THEN 'missing-output' END)`, id, state); err != nil {
			t.Fatal(err)
		}
		if _, err = pool.Exec(ctx, `INSERT INTO audit_executions(lane_class,execution_id,audit_id,role,workflow_role,submission_key,request_digest,state,terminal_outcome,terminal_observed_at,collection_receipt_id,collection_disposition,collection_request_digest,collection_retained_refs,collected_at)
VALUES('model',$1,$1,'task','inspect',$1,'sha256:'||repeat('e',64),'collected','submission-failed',clock_timestamp(),$1,'execution-failed','sha256:'||repeat('f',64),'[]',clock_timestamp())`, id); err != nil {
			t.Fatal(err)
		}
		if _, err = pool.Exec(ctx, `INSERT INTO audit_execution_items(execution_item_id,execution_id,audit_id,task_id,batch_ordinal,item_attempt,input_refs,state,collection_disposition,collected_at)
VALUES($1,$1,$1,$1,0,1,'[]','settled','execution-failed',clock_timestamp())`, id); err != nil {
			t.Fatal(err)
		}
	}
	snapshot := func() string {
		t.Helper()
		var s string
		if err := pool.QueryRow(ctx, `SELECT jsonb_build_object(
 'audits',(SELECT jsonb_agg(to_jsonb(a)-'phase'-'current_round_id' ORDER BY audit_id) FROM audits a),
 'tasks',(SELECT jsonb_agg(to_jsonb(t) ORDER BY task_id) FROM audit_tasks t),
 'executions',(SELECT jsonb_agg(to_jsonb(x)-ARRAY['role','round_id','role_attempt','manifest_ref','manifest_digest','preparation_snapshot','preparation_outputs'] ORDER BY execution_id) FROM audit_executions x),
 'attempts',(SELECT jsonb_agg(to_jsonb(m)-ARRAY['round_id','item_id','task_ref','task_digest','input_refs','finding_sources'] ORDER BY execution_item_id) FROM audit_execution_items m),
 'blobs',(SELECT jsonb_agg(to_jsonb(b) ORDER BY sha256) FROM artifact_blobs b),
 'revisions',(SELECT jsonb_agg(to_jsonb(r) ORDER BY scope_id,namespace,name,revision) FROM artifact_binding_revisions r),
 'links',(SELECT jsonb_agg(to_jsonb(l) ORDER BY audit_id,logical_key) FROM audit_artifact_links l))::text`).Scan(&s); err != nil {
			t.Fatal(err)
		}
		return s
	}
	before := snapshot()
	if _, err = applyMigrations(ctx, pool, MigrationBudgets{}, migrations); err != nil {
		t.Fatal(err)
	}
	if snapshot() != before {
		t.Fatal("retirement rewrote current work, original receipt, or exact retained bytes")
	}
	for _, name := range []string{"audit_items", "audit_rounds", "audit_round_result_pages", "audit_report_pages"} {
		var absent bool
		if err = pool.QueryRow(ctx, `SELECT to_regclass($1) IS NULL`, name).Scan(&absent); err != nil || !absent {
			t.Fatalf("retired relation %s survives: %v", name, err)
		}
	}
	if _, err = pool.Exec(ctx, `UPDATE audit_executions SET workflow_role='changed' WHERE execution_id='current-active'`); err == nil {
		t.Fatal("common execution identity became mutable")
	}
	if _, err = pool.Exec(ctx, `UPDATE audit_execution_items SET item_attempt=2 WHERE execution_item_id='current-active'`); err == nil {
		t.Fatal("attempt identity became mutable")
	}
	if err = VerifyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresAuditPipelineRetirementRefusesUnarchivedWorkAtomically(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	for _, state := range []string{"active", "paused", "completed"} {
		t.Run(state, func(t *testing.T) {
			ctx := t.Context()
			pool, _ := isolatedPools(t, ctx, dsn)
			migrations, err := loadMigrations()
			if err != nil {
				t.Fatal(err)
			}
			if _, err = applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:11]); err != nil {
				t.Fatal(err)
			}
			if _, err = pool.Exec(ctx, auditScopeMigrationFixtureSQL); err != nil {
				t.Fatal(err)
			}
			if _, err = pool.Exec(ctx, `UPDATE audits SET baseline_snapshot='{}',state=$1,started_at=clock_timestamp(),finished_at=CASE WHEN $1='completed' THEN clock_timestamp() END WHERE audit_id='scope-audit'`, state); err != nil {
				t.Fatal(err)
			}
			if _, err = pool.Exec(ctx, `INSERT INTO audit_rounds(round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count) VALUES('old-round','scope-audit',1,'{"namespace":"history","name":"original","revision":"exact"}','sha256:'||repeat('a',64),'closed',0)`); err != nil {
				t.Fatal(err)
			}
			var before, after string
			if err = pool.QueryRow(ctx, `SELECT to_jsonb(r)::text FROM audit_rounds r`).Scan(&before); err != nil {
				t.Fatal(err)
			}
			if _, err = applyMigrations(ctx, pool, MigrationBudgets{}, migrations); err == nil || !strings.Contains(err.Error(), "unarchived Audit pipeline records") {
				t.Fatalf("old history was not fenced: %v", err)
			}
			if err = pool.QueryRow(ctx, `SELECT to_jsonb(r)::text FROM audit_rounds r`).Scan(&after); err != nil || before != after {
				t.Fatalf("blocked migration changed history: %v", err)
			}
		})
	}
}
