package postgres

import (
	"os"
	"testing"
)

// Retained terminal attempts include every decision kind, exhausted escalation,
// and a manual target that has already terminated. Original receipt timestamps
// deliberately differ from Stage creation times and must not be reconstructed.
const stageOwnedStateFixtureSQL = `
INSERT INTO stage_executions(stage_execution_id,run_id,stage_name,attempt,previous_execution_id,
 resume_source_execution_id,stage_spec_schema_version,stage_spec_snapshot,
 stage_context_schema_version,stage_context_snapshot,state,state_reason_code,
 termination_schema_version,stage_termination,abort_id,abort_deadline,terminal_at)
SELECT id,'history-run',name,attempt,previous,manual,'v1','{}','v1','{}','interrupted','fixture',
 'v1','{"outcome":"interrupted","code":"fixture","message":"fixture","retryable":true,"phase":"preparing","occurredAt":"2026-10-01T12:00:00Z"}',
 id||'-abort',timestamptz '2026-10-01 12:01:00+00',timestamptz '2026-10-01 12:00:00+00'
FROM (VALUES
 ('next-source','next',1,NULL::text,NULL::text),('next-target','following',1,NULL,NULL),
 ('retry-source','retry',1,NULL,NULL),('retry-target','retry',2,'retry-source',NULL),
 ('escalate-source','escalate',1,NULL,NULL),('escalate-target','escalate',2,'escalate-source',NULL),
 ('success-source','success',1,NULL,NULL),('fail-source','fail',1,NULL,NULL),
 ('exhausted-source','exhausted',1,NULL,NULL),
 ('manual-source','manual',1,NULL,NULL),('manual-target','manual',2,'manual-source','manual-source')
) f(id,name,attempt,previous,manual);
INSERT INTO stage_transition_decisions(source_execution_id,run_id,action,target_stage_name,target_execution_id,
 escalation_ordinal,escalation_exhausted,decided_at)
SELECT source,'history-run',action,target_name,target,ordinal,exhausted,timestamptz '2026-10-01 12:00:17+00'
FROM (VALUES
 ('next-source','next','following','next-target',NULL::int,false),
 ('retry-source','retry','retry','retry-target',NULL,false),
 ('escalate-source','escalate','escalate','escalate-target',1,false),
 ('success-source','succeed',NULL,NULL,NULL,false),
 ('fail-source','fail',NULL,NULL,NULL,false),
 ('exhausted-source','fail',NULL,NULL,2,true),
 ('manual-source','fail',NULL,NULL,NULL,false)
) f(source,action,target_name,target,ordinal,exhausted);
INSERT INTO run_stage_resumptions(source_execution_id,run_id,target_execution_id,requested_by,requested_at)
VALUES('manual-source','history-run','manual-target','owner',timestamptz '2026-10-02 13:14:15+00');
`

func TestStageOwnedStateMigrationPreservesHistory(t *testing.T) {
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
	if len(migrations) < 16 || migrations[15].name != "000016_stage_owned_state.sql" {
		t.Fatal("missing Stage ownership migration")
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:15]); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO workflow_runs(run_id,owner_id,workflow_name,workflow_version,workflow_schema_version,workflow_snapshot,parameters,runtime_labels,runtime_config_snapshot,scheduler_lane_class,state,state_reason_code)
VALUES('history-run','owner','workflow','1','v1','{}','{}','{}',$1::jsonb,'model','initializing','fixture')`, builtInRunRuntimeSnapshot); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, stageOwnedStateFixtureSQL); err != nil {
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
 'runs',(SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM workflow_runs r),
 'stages',(SELECT jsonb_agg(to_jsonb(s) ORDER BY stage_execution_id) FROM stage_executions s),
 'decisions',(SELECT jsonb_agg(to_jsonb(d) ORDER BY source_execution_id) FROM stage_transition_decisions d),
 'receipts',(SELECT jsonb_agg(to_jsonb(r) ORDER BY source_execution_id) FROM run_stage_resumptions r),
 'events',(SELECT jsonb_agg(to_jsonb(e) ORDER BY run_id,sequence_number) FROM workflow_run_events e))::text`)
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:16]); err != nil {
		t.Fatal(err)
	}
	after := snapshot(`SELECT jsonb_build_object(
 'runs',(SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM workflow_runs r),
 'stages',(SELECT jsonb_agg(to_jsonb(s)-ARRAY['transition_action','transition_target_stage_name','transition_target_execution_id','transition_escalation_ordinal','transition_escalation_exhausted','transition_decided_at','resume_requested_by','resume_requested_at','resume_transaction_id'] ORDER BY stage_execution_id) FROM stage_executions s),
 'decisions',(SELECT jsonb_agg(jsonb_build_object('source_execution_id',stage_execution_id,'run_id',run_id,'action',transition_action,'target_stage_name',transition_target_stage_name,'target_execution_id',transition_target_execution_id,'escalation_ordinal',transition_escalation_ordinal,'escalation_exhausted',transition_escalation_exhausted,'decided_at',transition_decided_at) ORDER BY stage_execution_id) FROM stage_executions WHERE transition_action IS NOT NULL),
 'receipts',(SELECT jsonb_agg(jsonb_build_object('source_execution_id',resume_source_execution_id,'run_id',run_id,'target_execution_id',stage_execution_id,'requested_by',resume_requested_by,'requested_at',resume_requested_at,'transaction_id',resume_transaction_id) ORDER BY resume_source_execution_id) FROM stage_executions WHERE resume_source_execution_id IS NOT NULL),
 'events',(SELECT jsonb_agg(to_jsonb(e) ORDER BY run_id,sequence_number) FROM workflow_run_events e))::text`)
	if before != after {
		t.Fatalf("migration changed durable history:\nbefore %s\nafter %s", before, after)
	}
	if absent := snapshot(`SELECT (to_regclass('run_stage_resumptions') IS NULL AND to_regclass('stage_transition_decisions') IS NULL)::text`); absent != "true" {
		t.Fatal("dependent tables remain")
	}
	for _, statement := range []string{
		`UPDATE stage_executions SET transition_action='succeed' WHERE stage_execution_id='manual-source'`,
		`UPDATE stage_executions SET transition_decided_at=clock_timestamp() WHERE stage_execution_id='manual-source'`,
		`UPDATE stage_executions SET resume_transaction_id=pg_current_xact_id() WHERE stage_execution_id='manual-target'`,
		`UPDATE stage_executions SET resume_requested_at=clock_timestamp() WHERE stage_execution_id='manual-target'`,
	} {
		if _, err := pool.Exec(ctx, statement); SQLState(err) != SQLStateCheckViolation {
			t.Fatalf("retained history was mutable: %s: %v", statement, err)
		}
	}
	if err := VerifyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
}
