package postgres

import (
	"context"
	"testing"
	"time"
)

func TestPostgresAuditContinuationRemovalPreservesRetainedState(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 62)
	if _, err := pool.Exec(ctx, `
INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES ('audit-project','owner','project','Retained audits','create-project','sha256:'||repeat('a',64));
INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,
profile_snapshot,input_selection,baseline_snapshot,state,dispatch_state,hold_state,
max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes,
started_at,deadline_at,paused_at,finished_at,stop_reason_code,stop_reason_message,continuation_count)
SELECT 'audit-'||s.state,'owner','audit-project','profile','1','sha256:'||repeat('b',64),
'{}','{}','{}',s.state,CASE WHEN s.state='paused' THEN 'open' ELSE 'closed' END,
CASE WHEN s.state='paused' THEN 'held' ELSE 'released' END,1,1,1,1,1,1,1024,
'2026-09-01T10:00:00Z'::timestamptz,'2026-09-01T11:00:00Z'::timestamptz,
CASE WHEN s.state='paused' THEN '2026-09-01T11:00:00Z'::timestamptz END,
CASE WHEN s.state<>'paused' THEN '2026-09-01T11:00:00Z'::timestamptz END,
'deadline_exhausted','Time limit reached',s.continuations
FROM (VALUES ('paused',0),('completed',2),('failed',1)) s(state,continuations);
INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance)
VALUES ('audit-completed','report/machine','{"namespace":"audit-completed","name":"report-continuation-2.json","revision":"r1"}',
'sha256:'||repeat('c',64),'application/json',1,'{}'),
('audit-completed','report/history/3/machine','{"namespace":"audit-completed","name":"report.json","revision":"r1"}',
'sha256:'||repeat('d',64),'application/json',1,'{}');
`); err != nil {
		t.Fatal(err)
	}
	const snapshot = `SELECT jsonb_build_object(
'audits', (SELECT jsonb_agg(to_jsonb(a)-'continuation_count' ORDER BY audit_id) FROM audits a),
'reports', (SELECT jsonb_agg(to_jsonb(l) ORDER BY audit_id,logical_key) FROM audit_artifact_links l)
)::text`
	var before, after string
	if err := pool.QueryRow(ctx, snapshot).Scan(&before); err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(ctx, pool)
	if err != nil || len(result.AppliedVersions) == 0 || result.AppliedVersions[0] != 63 {
		t.Fatalf("remove continuation counter = %+v, %v", result, err)
	}
	if err := pool.QueryRow(ctx, snapshot).Scan(&after); err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatal("migration changed retained Audit state, pause clocks or report references")
	}
	var remains bool
	if err := pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='audits' AND column_name='continuation_count')`).Scan(&remains); err != nil {
		t.Fatal(err)
	}
	if remains {
		t.Fatal("obsolete continuation counter remains in the schema")
	}
	if replay, err := ApplyMigrations(ctx, pool); err != nil || len(replay.AppliedVersions) != 0 {
		t.Fatalf("migration replay = %+v, %v", replay, err)
	}
}
