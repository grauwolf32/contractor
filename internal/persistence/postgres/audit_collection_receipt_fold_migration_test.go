package postgres

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestPostgresAuditCollectionReceiptFoldMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 93)
	if _, err := pool.Exec(ctx, `
INSERT INTO projects (project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES ('upgrade-project','owner','project','Upgrade','upgrade','sha256:'||repeat('a',64));
INSERT INTO audits (
    audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,
    profile_snapshot,input_selection,max_rounds,batch_size,max_items_per_round,
    max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes
) VALUES ('audit','owner','upgrade-project','profile','1','sha256:'||repeat('a',64),
          '{}','{}',1,1,10,10,10,1,1024);
INSERT INTO audit_executions (
    execution_id, audit_id, role, role_attempt, workflow_role, manifest_ref, manifest_digest,
    submission_key, request_digest, state, terminal_outcome, terminal_observed_at
) VALUES ('execution', 'audit', 'discovery', 1, 'discovery',
          '{"namespace":"audit-executions","name":"discovery","revision":"r1"}'::jsonb,
          'sha256:'||repeat('a',64), 'submission', 'sha256:'||repeat('b',64),
          'collected', 'submission-failed', clock_timestamp());
INSERT INTO audit_collection_receipts (
    receipt_id, audit_id, execution_id, terminal_outcome, disposition,
    error_code, request_digest
) VALUES ('receipt', 'audit', 'execution', 'submission-failed', 'execution-failed',
          'submission_failed', 'sha256:'||repeat('c',64));
`); err != nil {
		t.Fatal(err)
	}

	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 94) {
		t.Fatalf("fold Audit collection receipts = %+v, %v", result, err)
	}
	var receiptID, disposition, errorCode string
	var tableGone bool
	if err := pool.QueryRow(ctx, `
SELECT collection_receipt_id, collection_disposition, collection_error_code,
       to_regclass('audit_collection_receipts') IS NULL
  FROM audit_executions
 WHERE execution_id = 'execution' AND collection_retained_refs = '[]'::jsonb
   AND collection_request_digest = 'sha256:'||repeat('c',64) AND collected_at IS NOT NULL`,
	).Scan(&receiptID, &disposition, &errorCode, &tableGone); err != nil {
		t.Fatal(err)
	}
	if receiptID != "receipt" || disposition != "execution-failed" || errorCode != "submission_failed" || !tableGone {
		t.Fatalf("collection = (%q, %q, %q), receipt table dropped %t", receiptID, disposition, errorCode, tableGone)
	}
	var pgErr *pgconn.PgError
	for _, statement := range []string{
		`UPDATE audit_executions SET collection_disposition = 'missing-output'`,
		`UPDATE audit_executions SET terminal_outcome = 'failed'`,
	} {
		_, err = pool.Exec(ctx, statement)
		if !errors.As(err, &pgErr) || pgErr.Code != "23514" {
			t.Fatalf("collected execution changed by %s: %v", statement, err)
		}
	}
	if _, err := pool.Exec(ctx, `UPDATE audit_executions SET updated_at = clock_timestamp()`); err != nil {
		t.Fatalf("collected execution bookkeeping rejected: %v", err)
	}
}
