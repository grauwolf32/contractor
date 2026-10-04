package postgres

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestPostgresFindingProposalRetentionFoldMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 92)
	if _, err := pool.Exec(ctx, `
INSERT INTO projects (project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES ('upgrade-project','owner','project','Upgrade','upgrade','sha256:'||repeat('a',64));
INSERT INTO finding_proposal_receipts (
    receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,
    stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,
    request_digest,run_id,owner_id,project_id,workflow_name,workflow_version,
    workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,
    proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence
) VALUES (
    'kept-receipt','proposal','allocation','runtime','instance','stage','worker',
    'invocation','submission','candidate','sha256:'||repeat('a',64),'deleted-run',
    'owner','upgrade-project','workflow','1','contractor/v1alpha1','{}',
    'sha256:'||repeat('a',64),'{"namespace":"findings","name":"proposal","revision":"r1"}',
    'sha256:'||repeat('a',64),'application/json',128,'[]'
);
INSERT INTO finding_proposal_retention(receipt_id,state,source_run_deleted_at,discarded_at)
VALUES ('kept-receipt','discarded',clock_timestamp(),clock_timestamp());
`); err != nil {
		t.Fatal(err)
	}

	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 93) {
		t.Fatalf("fold finding proposal retention = %+v, %v", result, err)
	}
	var state string
	var tableGone bool
	if err := pool.QueryRow(ctx, `
SELECT retention_state, to_regclass('finding_proposal_retention') IS NULL
  FROM finding_proposal_receipts
 WHERE receipt_id = 'kept-receipt' AND source_run_deleted_at IS NOT NULL AND discarded_at IS NOT NULL`,
	).Scan(&state, &tableGone); err != nil {
		t.Fatal(err)
	}
	if state != "discarded" || !tableGone {
		t.Fatalf("retention state=%q, side table dropped %t", state, tableGone)
	}
	var pgErr *pgconn.PgError
	for _, statement := range []string{
		`UPDATE finding_proposal_receipts SET proposal_media_type = 'text/plain'`,
		`DELETE FROM finding_proposal_receipts`,
	} {
		_, err = pool.Exec(ctx, statement)
		if !errors.As(err, &pgErr) || pgErr.Code != "23514" {
			t.Fatalf("receipt identity changed by %s: %v", statement, err)
		}
	}
	if _, err := pool.Exec(ctx, `
UPDATE finding_proposal_receipts SET retention_updated_at = clock_timestamp()`); err != nil {
		t.Fatalf("retention state update rejected: %v", err)
	}
}
