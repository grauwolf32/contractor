package postgres

import (
	"os"
	"testing"
)

func TestPostgresRemoveScanSweepsPreservesRunsAndSharedArtifacts(t *testing.T) {
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
	if len(available) < 2 || available[0].version != 1 || available[1].name != "000002_remove_scan_sweeps.sql" {
		t.Fatal("expected folded baseline followed by Sweep retirement")
	}
	previous := available[0].version
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:1]); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest) VALUES('scan-project','owner','project','Preserved','create','sha256:'||repeat('a',64))`); err != nil {
		t.Fatal(err)
	}
	for _, state := range []string{"pending", "succeeded"} {
		if _, err := pool.Exec(ctx, `INSERT INTO workflow_runs(run_id,owner_id,project_id,workflow_name,workflow_version,workflow_schema_version,workflow_snapshot,runtime_labels,runtime_config_snapshot,scheduler_lane_class,state,state_reason_code,finished_at,project_http_target_snapshot) VALUES($1,'owner','scan-project','nuclei-web','1','v1','{}','{}',$2::jsonb,'tool',$3,'test',CASE WHEN $3='succeeded' THEN clock_timestamp() END,'{"url":"https://original.example/"}')`, "child-"+state, builtInRunRuntimeSnapshot, state); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := pool.Exec(ctx, removeScanSweepFixtureSQL); err != nil {
		t.Fatal(err)
	}
	var before string
	// Compare all persisted child Run data, not only its identity or state.
	if err := pool.QueryRow(ctx, `SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id)::text FROM workflow_runs r`).Scan(&before); err != nil {
		t.Fatal(err)
	}
	result, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:2])
	if err != nil || len(result.AppliedVersions) == 0 || result.AppliedVersions[0] != previous+1 {
		t.Fatalf("migration=%+v err=%v", result, err)
	}
	var after string
	if err := pool.QueryRow(ctx, `SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id)::text FROM workflow_runs r`).Scan(&after); err != nil || after != before {
		t.Fatalf("child Runs changed: %v", err)
	}
	var absent bool
	if err := pool.QueryRow(ctx, `SELECT to_regclass('scan_sweeps') IS NULL AND to_regclass('scan_sweep_members') IS NULL AND to_regclass('projects_sweep_selection_idx') IS NULL AND to_regprocedure('contractor_scan_sweep_immutable()') IS NULL`).Scan(&absent); err != nil || !absent {
		t.Fatalf("sweep schema remains: %v", err)
	}
	for _, check := range []struct {
		sql  string
		want int
	}{
		{`SELECT count(*) FROM artifact_bindings WHERE namespace='contractor-system' AND name LIKE 'scan-sweep.%'`, 0},
		{`SELECT count(*) FROM artifact_versions WHERE version_id='private-version'`, 0},
		{`SELECT count(*) FROM artifact_blobs WHERE payload='private'::bytea`, 0},
		{`SELECT count(*) FROM artifact_lineage WHERE source_namespace='contractor-system' OR target_namespace='contractor-system'`, 0},
		{`SELECT count(*) FROM mutation_receipts WHERE key_scope[1]='scan-sweep'`, 0},
		{`SELECT count(*) FROM mutation_receipts WHERE key_scope[1]='project'`, 1},
		{`SELECT count(*) FROM audits`, 0},
		{`SELECT count(*) FROM projects WHERE project_id='scan-project'`, 1},
		{`SELECT count(*) FROM artifact_bindings`, 5},
		{`SELECT count(*) FROM artifact_lineage`, 1},
	} {
		var got int
		if err := pool.QueryRow(ctx, check.sql).Scan(&got); err != nil || got != check.want {
			t.Fatalf("%s: got %d want %d: %v", check.sql, got, check.want, err)
		}
	}
	var inputs, reports int
	if err := pool.QueryRow(ctx, `SELECT count(*) FILTER(WHERE r.namespace='inputs' AND b.payload='shared rules'::bytea),count(*) FILTER(WHERE r.namespace='reports' AND b.payload='report'::bytea) FROM artifact_binding_revisions r JOIN artifact_versions v USING(version_id) JOIN artifact_blobs b ON b.sha256=v.blob_sha256 WHERE r.scope_kind='run'`).Scan(&inputs, &reports); err != nil || inputs != 2 || reports != 1 {
		t.Fatalf("retained payloads=%d/%d: %v", inputs, reports, err)
	}
	// The removed receipt family and purge authority cannot be used again.
	if _, err := pool.Exec(ctx, `INSERT INTO mutation_receipts(key_scope,idempotency_key,operation,request_digest,resource_id,response,owner_id) VALUES(ARRAY['scan-sweep','owner','scan-sweep.create'],'new','scan-sweep.create','sha256:'||repeat('a',64),'new','{}','owner')`); err == nil {
		t.Fatal("removed receipt family accepted")
	}
	if err := pool.QueryRow(ctx, `SELECT set_config('contractor.lifecycle_purge','scan-sweep',false),contractor_lifecycle_purge_enabled()`).Scan(new(string), &absent); err != nil || absent {
		t.Fatalf("removed purge authority accepted: %v", err)
	}

}

// Both active and terminal parents, a pending member, two admitted children,
// shared source/input bytes and a private snapshot that has no other reader.
const removeScanSweepFixtureSQL = `
INSERT INTO scan_sweeps(sweep_id,owner_id,workflow,snapshot,snapshot_digest,state,total,pending,admitted,skipped,unsettled,settled_at)
VALUES('active','owner','nuclei-web@1','{}','sha256:'||repeat('a',64),'active',2,1,1,0,2,NULL),
 ('terminal','owner','nuclei-web@1','{}','sha256:'||repeat('a',64),'completed',1,0,1,0,0,clock_timestamp()),
 ('private','owner','nuclei-web@1','{}','sha256:'||repeat('a',64),'completed',1,0,0,1,0,clock_timestamp());
INSERT INTO scan_sweep_members(sweep_id,ordinal,project_id,target,child,child_key,child_digest,state,reason,run_id,settled)
VALUES('active',0,'scan-project','https://original.example/','{}','active-child','sha256:'||repeat('a',64),'admitted','','child-pending',false),
 ('active',1,'not-yet-dispatched','https://pending.example/','{}','pending-child','sha256:'||repeat('a',64),'pending','',NULL,false),
 ('terminal',0,'scan-project','https://original.example/','{}','terminal-child','sha256:'||repeat('a',64),'admitted','','child-succeeded',true);
INSERT INTO mutation_receipts(key_scope,idempotency_key,operation,request_digest,resource_id,response,owner_id)
SELECT ARRAY['scan-sweep','owner','scan-sweep.create'],id,'scan-sweep.create','sha256:'||repeat('a',64),id,'{}','owner' FROM unnest(ARRAY['active','terminal','purged']) id;
INSERT INTO mutation_receipts(key_scope,idempotency_key,operation,request_digest,resource_id,response,owner_id,project_id)
VALUES(ARRAY['project','owner','scan-project','scan-project','create'],'project','create','sha256:'||repeat('a',64),'scan-project','{}','owner','scan-project');
INSERT INTO artifact_scopes(scope_kind,scope_id) VALUES('user','owner'),('run','child-pending'),('run','child-succeeded');
INSERT INTO artifact_blobs(sha256,payload,size_bytes) SELECT sha256(data::bytea),data::bytea,octet_length(data) FROM unnest(ARRAY['shared rules','private','report']) data;
INSERT INTO artifact_versions(version_id,blob_sha256,media_type) VALUES('shared-version',sha256('shared rules'::bytea),'application/yaml'),('private-version',sha256('private'::bytea),'application/yaml'),('report-version',sha256('report'::bytea),'application/json');
INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id)
VALUES('user','owner','rules','source','r1','shared-version'),
 ('user','owner','contractor-system','scan-sweep.active.rulepack','r1','shared-version'),
 ('user','owner','contractor-system','scan-sweep.terminal.rulepack','r1','shared-version'),
 ('user','owner','contractor-system','scan-sweep.private.rulepack','r1','private-version'),
 ('user','owner','notes','scan-sweep.active.rulepack','r1','shared-version'),
 ('run','child-pending','inputs','rulepack','r1','shared-version'),
 ('run','child-succeeded','inputs','rulepack','r1','shared-version'),
 ('run','child-succeeded','reports','scan','r1','report-version');
INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes)
SELECT scope_kind,scope_id,namespace,name,revision,true,1,0 FROM artifact_binding_revisions;
INSERT INTO artifact_lineage(target_scope_kind,target_scope_id,target_namespace,target_name,target_revision,source_scope_kind,source_scope_id,source_namespace,source_name,source_revision,lineage_kind)
VALUES('user','owner','contractor-system','scan-sweep.active.rulepack','r1','user','owner','rules','source','r1','input_fork'),
 ('run','child-pending','inputs','rulepack','r1','user','owner','contractor-system','scan-sweep.active.rulepack','r1','input_fork'),
 ('run','child-succeeded','inputs','rulepack','r1','user','owner','rules','source','r1','input_fork');
`
