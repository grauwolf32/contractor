package postgres

import (
	"os"
	"slices"
	"strings"
	"testing"
)

// Migration 000042 copies each existing hold's receipt creation time into it,
// orders the Audit's hold index by that time, and from then on copies the
// time into every new hold and keeps it (V414-019, FND-R186).
func TestPostgresFindingHoldReceiptTimeMigrationBackfillsHolds(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 41)
	// An Audit's draft writes its Project's correlation fence (000041), which
	// each admitted hold advances.
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
 VALUES('hold-upgrade','owner','project','Holds','create','sha256:'||repeat('a',64));
 INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,profile_snapshot,input_selection,
 max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes)
 VALUES('hold-audit','owner','hold-upgrade','profile','1','sha256:'||repeat('a',64),'{}','{}',1,1,10,10,10,1,1024);
 INSERT INTO finding_correlation_fences(project_id) VALUES('hold-upgrade')`); err != nil {
		t.Fatal(err)
	}
	insertReceipt := func(receiptID, createdAt string) {
		t.Helper()
		if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_receipts(receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,
 stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,request_digest,run_id,owner_id,project_id,workflow_name,
 workflow_version,workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,proposal_digest,
 proposal_media_type,proposal_size_bytes,evidence,report_projection,created_at)
 VALUES($1,$1||'-proposal','allocation','runtime','instance','stage','worker','invocation',$1,$1,'sha256:'||repeat('a',64),'run',
 'owner','hold-upgrade','workflow','1','contractor/v1alpha1','{}','sha256:'||repeat('b',64),'{}','sha256:'||repeat('c',64),
 'application/json',2,'[]','{}',$2::timestamptz)`, receiptID, createdAt); err != nil {
			t.Fatal(err)
		}
	}
	insertHold := func(receiptID string) error {
		_, err := pool.Exec(ctx, `INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence)
 VALUES($1,'hold-audit','hold-upgrade',jsonb_build_object('name',$1::text),'[]')`, receiptID)
		return err
	}
	// The older receipt is held last, so hold time and receipt time disagree.
	insertReceipt("receipt-new", "2026-03-01T00:00:00Z")
	insertReceipt("receipt-old", "2026-01-01T00:00:00Z")
	for _, receiptID := range []string{"receipt-new", "receipt-old"} {
		if err := insertHold(receiptID); err != nil {
			t.Fatal(err)
		}
	}
	// Preserve this historical fixture before the incompatible required-conditions transition.
	applyMigrationsThrough(t, ctx, pool, "audit_admission_chain_transition")
	var mismatched, nullable int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM finding_proposal_audit_holds AS hold
          JOIN finding_proposal_receipts AS receipt USING (receipt_id)
         WHERE hold.receipt_created_at IS DISTINCT FROM receipt.created_at),
       (SELECT count(*) FROM information_schema.columns
         WHERE table_schema = current_schema() AND table_name = 'finding_proposal_audit_holds'
           AND column_name = 'receipt_created_at' AND is_nullable = 'YES')`).Scan(&mismatched, &nullable); err != nil {
		t.Fatal(err)
	}
	if mismatched != 0 || nullable != 0 {
		t.Fatalf("backfill left %d holds without their receipt time; nullable columns %d", mismatched, nullable)
	}
	rows, err := pool.Query(ctx, `
SELECT receipt_id FROM finding_proposal_audit_holds WHERE audit_id = 'hold-audit'
 ORDER BY receipt_created_at, receipt_id`)
	if err != nil {
		t.Fatal(err)
	}
	var order []string
	for rows.Next() {
		var receiptID string
		if err := rows.Scan(&receiptID); err != nil {
			t.Fatal(err)
		}
		order = append(order, receiptID)
	}
	if err := rows.Err(); err != nil || !slices.Equal(order, []string{"receipt-old", "receipt-new"}) {
		t.Fatalf("holds in receipt time order = (%v, %v)", order, err)
	}
	var definition string
	if err := pool.QueryRow(ctx, `
SELECT indexdef FROM pg_indexes
 WHERE schemaname = current_schema() AND tablename = 'finding_proposal_audit_holds'
   AND indexname = 'finding_proposal_audit_holds_audit_receipt_idx'`).Scan(&definition); err != nil {
		t.Fatal(err)
	}
	if want := "USING btree (audit_id, receipt_created_at, receipt_id)"; !strings.Contains(definition, want) {
		t.Fatalf("hold index = %q, want %q", definition, want)
	}
	var stale int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM pg_indexes
 WHERE schemaname = current_schema() AND indexname = 'finding_proposal_audit_holds_audit_idx'`).Scan(&stale); err != nil || stale != 0 {
		t.Fatalf("hold-time index left = (%d, %v)", stale, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE finding_proposal_audit_holds SET receipt_created_at = receipt_created_at + interval '1 day'`); Classify(err) != ClassRejected {
		t.Fatalf("hold receipt time update = %v, want a rejection", err)
	}
	insertReceipt("receipt-after", "2026-02-01T00:00:00Z")
	if err := insertHold("receipt-after"); err != nil {
		t.Fatal(err)
	}
	var copied bool
	if err := pool.QueryRow(ctx, `
SELECT hold.receipt_created_at = receipt.created_at
  FROM finding_proposal_audit_holds AS hold JOIN finding_proposal_receipts AS receipt USING (receipt_id)
 WHERE hold.receipt_id = 'receipt-after'`).Scan(&copied); err != nil || !copied {
		t.Fatalf("new hold receipt time copied = (%t, %v)", copied, err)
	}
}
