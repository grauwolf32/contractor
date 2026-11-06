package postgres

import (
	"os"
	"testing"
)

// Deliberately noncanonical document bytes and non-default revisions/timestamps
// make the backfill prove preservation, not merely reconstruct equivalent JSON.
const evalOwnedStateFixtureSQL = `
INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES('eval-project','owner','evaluation','Eval','create','sha256:'||repeat('a',64));
INSERT INTO eval_experiments(experiment_id,owner_id,project_id,portable_id,control_mode,name,state,expected_count,max_in_flight,wall_ms)
SELECT id,'owner','eval-project',id,mode,id,'ready',1,1,1000
FROM (VALUES('native','server'),('external','external')) f(id,mode);
INSERT INTO eval_frozen_plans(experiment_id,document_kind,document,plan_sha256,setup,readiness,created_at)
SELECT experiment_id,CASE WHEN control_mode='server' THEN 'playground.plan/v1' ELSE 'ExternalRegistration' END,
 convert_to(E'{ "pin" : 1 }\n','UTF8'),'sha256:'||repeat('b',64),convert_to('{ "setup" : true }','UTF8'),
 CASE WHEN control_mode='server' THEN convert_to('{ "ready" : true }','UTF8') END,
 timestamptz '2026-10-01 12:00:00+00'
FROM eval_experiments;
INSERT INTO eval_members(experiment_id,member_id,pair_id,ordinal,suite_id,case_id,sample,variant_id,case_sha256,binding_sha256,eligibility,execution_kind,recipe,submission_key)
SELECT experiment_id,repeat('a',64),repeat('b',64),0,'suite','case',1,'a','sha256:'||repeat('c',64),'sha256:'||repeat('d',64),'eligible','run','{}','key-'||experiment_id
FROM eval_experiments;
INSERT INTO eval_plan_resources(experiment_id,resource_path,document_kind,document)
SELECT experiment_id,'case.json','playground.case/v2',convert_to('{ "case" : true }','UTF8') FROM eval_experiments;
INSERT INTO eval_records(experiment_id,member_id,kind,record_sha256,document,actor_id)
SELECT experiment_id,member_id,kind,'sha256:'||encode(sha256(document),'hex'),document,'reviewer'
FROM eval_members CROSS JOIN (VALUES('result',convert_to('{ "result" : true }','UTF8')),('assessment',convert_to('{ "assessment" : true }','UTF8'))) f(kind,document);
INSERT INTO eval_selections(experiment_id,member_id,result_sha256,assessment_sha256,revision,actor_id)
SELECT experiment_id,member_id,'sha256:'||encode(sha256(convert_to('{ "result" : true }','UTF8')),'hex'),
 'sha256:'||encode(sha256(convert_to('{ "assessment" : true }','UTF8')),'hex'),9,'reviewer' FROM eval_members;
`

func TestEvalOwnedStateMigrationPreservesPlansAndSelections(t *testing.T) {
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
	if len(migrations) < 14 || migrations[13].name != "000014_eval_owned_state.sql" {
		t.Fatal("missing Eval ownership migration")
	}
	if _, err = applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:13]); err != nil {
		t.Fatal(err)
	}
	if _, err = pool.Exec(ctx, evalOwnedStateFixtureSQL); err != nil {
		t.Fatal(err)
	}
	snapshot := func(query string) string {
		t.Helper()
		var result string
		if err := pool.QueryRow(ctx, query).Scan(&result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	before := snapshot(`SELECT jsonb_build_object(
 'experiments',(SELECT jsonb_agg(to_jsonb(e) ORDER BY experiment_id) FROM eval_experiments e),
 'plans',(SELECT jsonb_agg(to_jsonb(p) ORDER BY experiment_id) FROM eval_frozen_plans p),
 'members',(SELECT jsonb_agg(to_jsonb(m) ORDER BY experiment_id,member_id) FROM eval_members m),
 'selections',(SELECT jsonb_agg(to_jsonb(s) ORDER BY experiment_id,member_id) FROM eval_selections s),
 'resources',(SELECT jsonb_agg(to_jsonb(r) ORDER BY experiment_id,resource_path) FROM eval_plan_resources r),
 'queue',(SELECT jsonb_agg(to_jsonb(q) ORDER BY experiment_id) FROM eval_projection_queue q),
 'projections',(SELECT jsonb_agg(to_jsonb(p) ORDER BY experiment_id,member_id) FROM eval_member_projections p))::text`)
	if _, err = applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:14]); err != nil {
		t.Fatal(err)
	}
	after := snapshot(`SELECT jsonb_build_object(
 'experiments',(SELECT jsonb_agg(to_jsonb(e)-ARRAY['plan_kind','plan_document','plan_document_sha256','plan_sha256','plan_setup','plan_created_at','plan_readiness'] ORDER BY experiment_id) FROM eval_experiments e),
 'plans',(SELECT jsonb_agg(jsonb_build_object('experiment_id',experiment_id,'document_kind',plan_kind,'document',plan_document,'document_sha256',plan_document_sha256,'plan_sha256',plan_sha256,'setup',plan_setup,'created_at',plan_created_at,'readiness',plan_readiness) ORDER BY experiment_id) FROM eval_experiments),
 'members',(SELECT jsonb_agg(to_jsonb(m)-ARRAY['selected_result_kind','selected_result_sha256','selected_assessment_kind','selected_assessment_sha256','selection_revision','selected_by'] ORDER BY experiment_id,member_id) FROM eval_members m),
 'selections',(SELECT jsonb_agg(jsonb_build_object('experiment_id',experiment_id,'member_id',member_id,'result_kind',selected_result_kind,'result_sha256',selected_result_sha256,'assessment_kind',selected_assessment_kind,'assessment_sha256',selected_assessment_sha256,'revision',selection_revision,'actor_id',selected_by) ORDER BY experiment_id,member_id) FROM eval_members WHERE selected_result_sha256 IS NOT NULL),
 'resources',(SELECT jsonb_agg(to_jsonb(r) ORDER BY experiment_id,resource_path) FROM eval_plan_resources r),
 'queue',(SELECT jsonb_agg(to_jsonb(q) ORDER BY experiment_id) FROM eval_projection_queue q),
 'projections',(SELECT jsonb_agg(to_jsonb(p) ORDER BY experiment_id,member_id) FROM eval_member_projections p))::text`)
	if after != before {
		t.Fatalf("migration changed private authority or projections:\nbefore %s\nafter %s", before, after)
	}
	if n := snapshot(`SELECT count(*)::text FROM information_schema.tables WHERE table_schema=current_schema() AND table_name LIKE 'eval\_%'`); n != "17" {
		t.Fatalf("Eval tables=%s, want 17", n)
	}
	for _, q := range []string{
		`UPDATE eval_experiments SET plan_document=convert_to('{}','UTF8'),updated_at=clock_timestamp() WHERE experiment_id='native'`,
		`UPDATE eval_members SET recipe=convert_to('{"changed":true}','UTF8') WHERE experiment_id='native'`,
	} {
		if _, err := pool.Exec(ctx, q); SQLState(err) != SQLStateCheckViolation {
			t.Fatalf("immutable fields accepted: %v", err)
		}
	}
	// Current selections remain editable, and exact records remain protected.
	if _, err := pool.Exec(ctx, `UPDATE eval_members SET selection_revision=selection_revision+1,selected_by='another-reviewer' WHERE experiment_id='native'`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE eval_members SET selected_result_sha256='sha256:'||repeat('f',64),selection_revision=selection_revision+1 WHERE experiment_id='native'`); SQLState(err) != SQLStateForeignKeyViolation {
		t.Fatalf("selection lost its exact record foreign key: %v", err)
	}
}
