package postgres

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"
	"time"
)

func TestPostgresAuditPreparationUpgradePreservesCurrentSnapshotsReceiptsAndHolds(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 97)
	// Frozen current-schema closure; validation is selected by the artifact
	// preparation suite, outside this package's PostgreSQL import boundary.
	raw, err := os.ReadFile("testdata/audit-profile-current.json")
	if err != nil {
		t.Fatal(err)
	}
	var profile struct {
		Ref struct{ Name, Version, Digest string }
	}
	if err := json.Unmarshal(raw, &profile); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO projects (project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES ('project','owner','project','Upgrade','upgrade','sha256:'||repeat('a',64));
`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO audits (
    audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,profile_snapshot,input_selection,
    max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes,
    baseline_snapshot,state,hold_state,started_at,deadline_at
) VALUES ('audit','owner','project',$1,$2,$3,$4,'{}',1,1,10,10,10,1,1024,
    '{"schema":"contractor.audit.baseline.v1","inputs":{},"llmCredentialIds":["credential"]}',
    'active','held',clock_timestamp(),clock_timestamp()+interval '1 hour')`, profile.Ref.Name, profile.Ref.Version, profile.Ref.Digest, raw); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO audit_rounds (round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count)
VALUES ('round','audit',1,'{"namespace":"audit-manifests","name":"inventory","revision":"r1"}',
        'sha256:'||repeat('a',64),'accepted',1);
UPDATE audits SET current_round_id='round' WHERE audit_id='audit';
INSERT INTO audit_executions (
    execution_id,audit_id,round_id,role,workflow_role,role_attempt,manifest_ref,manifest_digest,submission_key,request_digest,
    state,terminal_outcome,terminal_observed_at,collection_receipt_id,collection_disposition,collection_retained_refs,
    collection_error_code,collection_request_digest,collected_at
) VALUES ('execution','audit','round','discovery','discover',1,
    '{"namespace":"audit-manifests","name":"execution","revision":"r1"}','sha256:'||repeat('b',64),
    'submission','sha256:'||repeat('c',64),'collected','submission-failed',clock_timestamp(),'receipt','execution-failed',
    '[{"logicalKey":"diagnostics","artifact":{"ref":{"namespace":"audit-retained","name":"diagnostic","revision":"r1"},"digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","mediaType":"application/json","sizeBytes":1},"sourceProvenance":{}}]',
    'submission_failed','sha256:'||repeat('d',64),clock_timestamp());
INSERT INTO audit_artifact_links (audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance)
VALUES ('audit','diagnostics','{"namespace":"audit-retained","name":"diagnostic","revision":"r1"}',
        'sha256:'||repeat('d',64),'application/json',1,'{}');
`); err != nil {
		t.Fatal(err)
	}
	const snapshot = `SELECT jsonb_build_object(
    'audits',(SELECT jsonb_agg(to_jsonb(a)-'phase' ORDER BY audit_id) FROM audits a),
    'rounds',(SELECT jsonb_agg(to_jsonb(r) ORDER BY round_id) FROM audit_rounds r),
    'executions',(SELECT jsonb_agg(to_jsonb(e)-ARRAY['preparation_snapshot','preparation_outputs'] ORDER BY execution_id) FROM audit_executions e),
    'links',(SELECT jsonb_agg(to_jsonb(l) ORDER BY logical_key) FROM audit_artifact_links l),
    'events',(SELECT jsonb_agg(to_jsonb(e) ORDER BY sequence_number) FROM audit_events e)
)::text`
	var before, after string
	if err := pool.QueryRow(ctx, snapshot).Scan(&before); err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !reflect.DeepEqual(result.AppliedVersions, []int64{98}) {
		t.Fatalf("preparation upgrade: %+v %v", result, err)
	}
	if err := pool.QueryRow(ctx, snapshot).Scan(&after); err != nil || before != after {
		t.Fatalf("upgrade changed retained authority: %v", err)
	}
	var phase string
	var profileAfter []byte
	if err := pool.QueryRow(ctx, `SELECT phase,profile_snapshot FROM audits WHERE audit_id='audit'`).Scan(&phase, &profileAfter); err != nil {
		t.Fatal(err)
	}
	if phase != "rounds" {
		t.Fatalf("backfilled phase: %s", phase)
	}
	var profilePreserved bool
	if err := pool.QueryRow(ctx, `SELECT profile_snapshot=$1::jsonb FROM audits WHERE audit_id='audit'`, raw).Scan(&profilePreserved); err != nil || !profilePreserved {
		t.Fatalf("profile snapshot changed: %v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audit_executions SET collection_request_digest='sha256:'||repeat('e',64)`); SQLState(err) != "23514" {
		t.Fatalf("receipt lost immutability: %v", err)
	}
	if result, err := ApplyMigrations(ctx, pool); err != nil || len(result.AppliedVersions) != 0 {
		t.Fatalf("migration replay: %+v %v", result, err)
	}
}
