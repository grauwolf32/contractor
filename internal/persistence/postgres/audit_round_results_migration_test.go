package postgres

import (
	"bytes"
	"os"
	"testing"
)

func TestPostgresRoundResultsMigrationPreservesBusinessRows(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "audit_round_results")
	installMigrationPrefix(t, ctx, pool, previous)
	for _, sql := range []string{
		`INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES('snapshot-migration-project','snapshot-owner','project','Snapshot migration','snapshot-project','sha256:'||repeat('a',64))`,
		`INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,
profile_snapshot,input_selection,baseline_snapshot,state,phase,max_rounds,batch_size,max_items_per_round,max_items_total,
max_submitted_runs,max_item_run_attempts,max_evidence_bytes,started_at)
VALUES('snapshot-migration-audit','snapshot-owner','snapshot-migration-project','snapshot-profile','1','sha256:'||repeat('b',64),
'{"workflows":{"check":{"kind":"check"}}}','{"inputs":{}}','{"schema":"contractor.audit.baseline.v1","inputs":{},"skills":[]}',
'active','rounds',1,1,1,1,1,1,67108864,clock_timestamp())`,
		`INSERT INTO audit_rounds(round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count)
VALUES('snapshot-migration-round','snapshot-migration-audit',1,'{"namespace":"audit-snapshot","name":"manifest","revision":"r1"}',
'sha256:'||repeat('c',64),'assessing',1)`,
		`INSERT INTO audit_items(item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,task_ref,task_digest,
origin,workflow_role,state,final_disposition,coverage_status,coverage_requested,coverage_completed,coverage_gaps,coverage_rationale)
VALUES('snapshot-migration-item','snapshot-migration-audit','snapshot-migration-round','item',0,'checklist','subject',
'{"namespace":"audit-snapshot","name":"task","revision":"r1"}','sha256:'||repeat('d',64),
jsonb_build_object('schema','contractor.audit.item-origin.v1','entryKey','item',
'sourceRef',jsonb_build_object('namespace','fixture','name','source','revision','r1'),
'sourceContentDigest','sha256:'||repeat('e',64),'canonicalInventoryDigest','sha256:'||repeat('f',64),'sourceMediaType','application/json'),
'check','settled','excluded','excluded','["code-review"]','[]','["not-tested"]','Retained fixture rationale')`,
	} {
		if _, err := pool.Exec(ctx, sql); err != nil {
			t.Fatal(err)
		}
	}
	var before []byte
	if err := pool.QueryRow(ctx, `SELECT jsonb_build_array(audit.profile_snapshot,audit.baseline_snapshot,
round.manifest_ref,item.task_ref,item.origin,item.coverage_requested,item.coverage_completed,item.coverage_gaps,item.coverage_rationale)::text
FROM audits AS audit JOIN audit_rounds AS round USING(audit_id) JOIN audit_items AS item USING(audit_id,round_id)
WHERE audit.audit_id='snapshot-migration-audit'`).Scan(&before); err != nil {
		t.Fatal(err)
	}
	applyMigrationsThrough(t, ctx, pool, "audit_round_results")
	var after []byte
	var generation *int64
	var sealed bool
	var count int
	var size int64
	if err := pool.QueryRow(ctx, `SELECT jsonb_build_array(audit.profile_snapshot,audit.baseline_snapshot,
round.manifest_ref,item.task_ref,item.origin,item.coverage_requested,item.coverage_completed,item.coverage_gaps,item.coverage_rationale)::text,
round.results_generation,round.results_sealed,round.results_item_count,item.results_projection_bytes
FROM audits AS audit JOIN audit_rounds AS round USING(audit_id) JOIN audit_items AS item USING(audit_id,round_id)
WHERE audit.audit_id='snapshot-migration-audit'`).Scan(&after, &generation, &sealed, &count, &size); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) || generation != nil || sealed || count != 0 || size < 4096 {
		t.Fatalf("migration rewrote business content or invented a snapshot: gen=%v sealed=%t count=%d size=%d", generation, sealed, count, size)
	}
}
