package postgres

import (
	"os"
	"testing"
)

func TestPostgresDiscardedFindingReportMigrationPreservesReplayProvenance(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "discarded_finding_report_projection")
	installMigrationPrefix(t, ctx, pool, previous)
	const report = `{"title":"Retained issue","description":"Private finding details","standard_refs":[]}`
	insert := func(id, retention, projection string) error {
		_, err := pool.Exec(ctx, `INSERT INTO finding_proposal_receipts (
 receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,request_digest,run_id,owner_id,
 workflow_name,workflow_version,workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence,report_projection,
 retention_state,source_run_deleted_at,discarded_at)
 VALUES($1,$1,$1,'runtime','instance','stage','worker','invocation',$1,$1,'sha256:'||repeat('a',64),'run','owner',
 'workflow','1','contractor/v1alpha1','{}','sha256:'||repeat('b',64),'{"namespace":"finding-proposals","name":"candidate","revision":"1"}','sha256:'||repeat('c',64),'application/json',128,
 '[{"ref":{"namespace":"worker","name":"trace","revision":"1"},"digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","mediaType":"text/plain","sizeBytes":42}]',$3,
 $2,CASE WHEN $2='source-held' THEN NULL ELSE clock_timestamp() END,CASE WHEN $2='discarded' THEN clock_timestamp() ELSE NULL END)`, id, retention, projection)
		return err
	}
	for _, state := range []string{"source-held", "audit-held", "discarded"} {
		if err := insert(state, state, report); err != nil {
			t.Fatal(err)
		}
	}
	// A snapshot of every other column proves the upgrade changes no replay
	// identity, exact descriptor, state, timestamp or source provenance.
	const provenanceSQL = `SELECT jsonb_agg(to_jsonb(r)-ARRAY['report_projection','fingerprint_projection','anchor_projection','comparison_projection'] ORDER BY receipt_id)::text FROM finding_proposal_receipts r`
	var before, after string
	if err := pool.QueryRow(ctx, provenanceSQL).Scan(&before); err != nil {
		t.Fatal(err)
	}
	applyMigrationsThrough(t, ctx, pool, "discarded_finding_report_projection")
	if err := pool.QueryRow(ctx, provenanceSQL).Scan(&after); err != nil {
		t.Fatal(err)
	}
	if after != before {
		t.Fatal("discard migration changed receipt identity or bounded provenance")
	}
	for _, state := range []string{"source-held", "audit-held", "discarded"} {
		want := report
		if state == "discarded" {
			want = "{}"
		}
		var got string
		if err := pool.QueryRow(ctx, "SELECT report_projection FROM finding_proposal_receipts WHERE receipt_id=$1", state).Scan(&got); err != nil || got != want {
			t.Fatalf("%s report projection = %q, %v; want %q", state, got, err, want)
		}
		if state != "discarded" {
			if _, err := pool.Exec(ctx, "UPDATE finding_proposal_receipts SET report_projection='{}' WHERE receipt_id=$1", state); Classify(err) != ClassRejected {
				t.Fatalf("%s report mutation = %v, want immutable rejection", state, err)
			}
		}
	}
	// Neither backfilled nor new tombstones may acquire report content again.
	if _, err := pool.Exec(ctx, "UPDATE finding_proposal_receipts SET report_projection=$1 WHERE receipt_id='discarded'", report); err != nil {
		t.Fatal(err)
	}
	var got string
	if err := pool.QueryRow(ctx, "SELECT report_projection FROM finding_proposal_receipts WHERE receipt_id='discarded'").Scan(&got); err != nil || got != "{}" {
		t.Fatalf("discarded report restored: %q, %v", got, err)
	}
	if err := insert("forged-discard", "discarded", report); Classify(err) != ClassInvalid || ConstraintName(err) != "finding_proposal_receipts_discarded_report_check" {
		t.Fatalf("new discarded report content = %v, want constraint rejection", err)
	}
}
