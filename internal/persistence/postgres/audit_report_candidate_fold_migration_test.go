package postgres

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestPostgresAuditReportCandidateFoldMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 95)
	if _, err := pool.Exec(ctx, `
INSERT INTO projects (project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES ('upgrade-project','owner','project','Upgrade','upgrade','sha256:'||repeat('a',64));
INSERT INTO audits (
    audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,
    profile_snapshot,input_selection,max_rounds,batch_size,max_items_per_round,
    max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes
) VALUES ('audit','owner','upgrade-project','profile','1','sha256:'||repeat('a',64),
          '{}','{}',1,1,10,10,10,1,1024);
INSERT INTO audit_rounds (round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count)
VALUES ('round','audit',1,'{"namespace":"audit-rounds","name":"round","revision":"r1"}','sha256:'||repeat('a',64),'closed',0);
INSERT INTO audit_review_requests (
    request_id,audit_id,finding_id,subject_kind,subject_id,kind,subject_revision,subject_digest,
    requested_actions,state,expires_at,idempotency_key,request_digest
) VALUES ('report-review','audit',NULL,'audit-report','audit','report-acceptance',3,'sha256:'||repeat('c',64),
          '["approve","reject"]','pending',clock_timestamp() + interval '30 days','auto-report','sha256:'||repeat('c',64));
INSERT INTO audit_report_candidates (
    audit_id,request_id,round_id,subject_revision,subject_digest,machine_link,summary_link
) VALUES ('audit','report-review','round',3,'sha256:'||repeat('c',64),'{"logicalKey":"report/machine"}','{"logicalKey":"report/summary"}');
`); err != nil {
		t.Fatal(err)
	}

	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 96) {
		t.Fatalf("fold Audit report candidates = %+v, %v", result, err)
	}
	var roundID, machineKey, summaryKey string
	var tableGone bool
	if err := pool.QueryRow(ctx, `
SELECT report_round_id, report_machine_link->>'logicalKey', report_summary_link->>'logicalKey',
       to_regclass('audit_report_candidates') IS NULL
  FROM audit_review_requests WHERE request_id = 'report-review'`,
	).Scan(&roundID, &machineKey, &summaryKey, &tableGone); err != nil {
		t.Fatal(err)
	}
	if roundID != "round" || machineKey != "report/machine" || summaryKey != "report/summary" || !tableGone {
		t.Fatalf("report review = (%q, %q, %q), candidate table dropped %t", roundID, machineKey, summaryKey, tableGone)
	}
	var pgErr *pgconn.PgError
	_, err = pool.Exec(ctx, `
INSERT INTO audit_review_requests (
    request_id,audit_id,finding_id,subject_kind,subject_id,kind,subject_revision,subject_digest,
    requested_actions,state,idempotency_key,request_digest,
    report_round_id,report_machine_link,report_summary_link
) VALUES ('second-report','audit',NULL,'audit-report','audit','report-acceptance',4,'sha256:'||repeat('d',64),
          '["approve","reject"]','pending','second','sha256:'||repeat('d',64),'round','{}','{}')`)
	if !errors.As(err, &pgErr) || pgErr.Code != "23505" {
		t.Fatalf("second report candidate for one Audit: %v", err)
	}
}
