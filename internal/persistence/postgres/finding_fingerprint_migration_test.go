package postgres

import (
	"os"
	"testing"
)

func TestPostgresFindingFingerprintMigrationPreservesReceipts(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 21)
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_receipts(receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,request_digest,run_id,owner_id,workflow_name,workflow_version,workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence,report_projection)
 VALUES('old-receipt','proposal','allocation','runtime','instance','stage','worker','invocation','submission','client','sha256:'||repeat('a',64),'run','owner','workflow','1','contractor/v1alpha1','{}','sha256:'||repeat('b',64),'{}','sha256:'||repeat('c',64),'application/json',2,'[]','{}')`); err != nil {
		t.Fatal(err)
	}
	var before, after, projection string
	if err := pool.QueryRow(ctx, "SELECT to_jsonb(r)::text FROM finding_proposal_receipts r").Scan(&before); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, "SELECT (to_jsonb(r)-'fingerprint_projection')::text,fingerprint_projection FROM finding_proposal_receipts r").Scan(&after, &projection); err != nil {
		t.Fatal(err)
	}
	if before != after || projection != `{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable","reason":"not_recorded"}` {
		t.Fatalf("migration changed receipt: before=%s after=%s fingerprint=%s", before, after, projection)
	}
	if _, err := pool.Exec(ctx, `UPDATE finding_proposal_receipts SET fingerprint_projection='{}'`); Classify(err) != ClassRejected {
		t.Fatalf("projection mutation=%v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE finding_proposal_receipts SET retention_state='discarded',source_run_deleted_at=clock_timestamp(),discarded_at=clock_timestamp()`); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, "SELECT fingerprint_projection FROM finding_proposal_receipts").Scan(&projection); err != nil {
		t.Fatal(err)
	}
	if projection != `{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable","reason":"discarded"}` {
		t.Fatalf("discard projection=%s", projection)
	}
}
