package postgres

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestPostgresAuditItemCoverageAndSourceFoldMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 94)
	if _, err := pool.Exec(ctx, `
INSERT INTO projects (project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES ('upgrade-project','owner','project','Upgrade','upgrade','sha256:'||repeat('a',64));
INSERT INTO audits (
    audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,
    profile_snapshot,input_selection,max_rounds,batch_size,max_items_per_round,
    max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes
) VALUES ('audit','owner','upgrade-project','profile','1','sha256:'||repeat('a',64),
          '{}','{}',2,1,10,10,10,1,1024);
INSERT INTO audit_rounds (round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count)
VALUES ('round','audit',2,'{"namespace":"audit-rounds","name":"round","revision":"r1"}','sha256:'||repeat('a',64),'accepted',1);
INSERT INTO audit_items (
    item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,
    task_ref,task_digest,origin,workflow_role,state
) VALUES ('item','audit','round','check',0,'check','subject',
          '{"namespace":"audit-task-packages","name":"check","revision":"r1"}','sha256:'||repeat('a',64),
          '{}','check-role','ready');
INSERT INTO audit_coverage_rows (
    audit_id,round_id,item_id,item_key,subject_key,status,requested,completed,gaps,rationale
) VALUES ('audit','round','item','check','subject','blocked','["a"]','[]','["gap"]','why');
INSERT INTO finding_proposal_receipts (
    receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,
    stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,
    request_digest,run_id,owner_id,project_id,workflow_name,workflow_version,
    workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,
    proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence
) VALUES (
    'receipt','proposal','allocation','runtime','instance','stage','worker',
    'invocation','submission','candidate','sha256:'||repeat('a',64),'run',
    'owner','upgrade-project','workflow','1','contractor/v1alpha1','{}',
    'sha256:'||repeat('a',64),'{"namespace":"findings","name":"proposal","revision":"r1"}',
    'sha256:'||repeat('a',64),'application/json',128,'[]'
);
INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence)
VALUES ('receipt','audit','upgrade-project','{}','[]');
INSERT INTO audit_proposal_items (
    audit_id,receipt_id,proposed_check_ordinal,round_id,item_id,proposal_ref,proposal_digest
) VALUES ('audit','receipt',3,'round','item','{"name":"proposal"}','sha256:'||repeat('b',64));
`); err != nil {
		t.Fatal(err)
	}

	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 95) {
		t.Fatalf("fold Audit item coverage and sources = %+v, %v", result, err)
	}
	var status, rationale, receiptID string
	var ordinal int
	var tablesGone bool
	if err := pool.QueryRow(ctx, `
SELECT coverage_status, coverage_rationale, proposal_receipt_id, proposal_check_ordinal,
       to_regclass('audit_coverage_rows') IS NULL AND to_regclass('audit_proposal_items') IS NULL
  FROM audit_items
 WHERE item_id = 'item' AND coverage_requested = '["a"]' AND coverage_gaps = '["gap"]'
   AND proposal_digest = 'sha256:'||repeat('b',64)`,
	).Scan(&status, &rationale, &receiptID, &ordinal, &tablesGone); err != nil {
		t.Fatal(err)
	}
	if status != "blocked" || rationale != "why" || receiptID != "receipt" || ordinal != 3 || !tablesGone {
		t.Fatalf("item = (%q, %q, %q, %d), side tables dropped %t", status, rationale, receiptID, ordinal, tablesGone)
	}
	var pgErr *pgconn.PgError
	_, err = pool.Exec(ctx, `UPDATE audit_items SET proposal_check_ordinal = 4`)
	if !errors.As(err, &pgErr) || pgErr.Code != "23514" {
		t.Fatalf("proposal source changed: %v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audit_items SET coverage_status = 'satisfied'`); err != nil {
		t.Fatalf("coverage update rejected: %v", err)
	}
}
