package postgres

import (
	"os"
	"strings"
	"testing"
)

func TestPostgresRemoveAuditApprovalsPreservesOpaqueArtifactsAndRestrictsPolicy(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, dsn)
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	const previous = 6
	if len(available) <= previous || available[previous].name != "000007_remove_audit_approvals.sql" {
		t.Fatal("missing approval-removal migration")
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:previous]); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, auditScopeMigrationFixtureSQL); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE projects SET active_checks='approval-required',revision=revision+1,updated_at=clock_timestamp() WHERE project_id='scope-project'`); err != nil {
		t.Fatal(err)
	}
	// These are already-archived facts from the existing cutover, not inputs
	// to this migration. Keep exact snapshots, decisions and report/result bytes.
	for name, content := range map[string]string{
		"old-snapshot":  `{"profile":{"interaction":{"reportAcceptance":"human-required"}},"state":"completed"}`,
		"old-decisions": `{"request_id":"historical-request","action":"approve","actor_id":"historical-owner","rationale":"Historical consent only"}`,
		"old-result":    `{"assessment":"inconclusive","limitations":["evidence missing"]}`,
		"old-report":    "# Historical report\n\nRetain these exact bytes.\n",
	} {
		body := []byte(content)
		if _, err := pool.Exec(ctx, `INSERT INTO artifact_blobs(sha256,payload,size_bytes) VALUES(sha256($1::bytea),$1,$2)`, body, len(body)); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `INSERT INTO artifact_versions(version_id,blob_sha256,media_type) VALUES($1,sha256($2::bytea),'application/octet-stream')`, name, body); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id) VALUES('project','scope-project','history',$1,'r1',$1)`, name); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes) VALUES('project','scope-project','history',$1,'r1',true,1,$2)`, name, len(body)); err != nil {
			t.Fatal(err)
		}
	}

	snapshot := func() string {
		t.Helper()
		var value string
		if err := pool.QueryRow(ctx, `SELECT jsonb_build_object('versions',(SELECT jsonb_agg(to_jsonb(v) ORDER BY version_id) FROM artifact_versions v),'revisions',(SELECT jsonb_agg(to_jsonb(r) ORDER BY revision) FROM artifact_binding_revisions r),'blobs',(SELECT jsonb_agg(to_jsonb(b) ORDER BY sha256) FROM artifact_blobs b),'bindings',(SELECT jsonb_agg(to_jsonb(b) ORDER BY namespace,name) FROM artifact_bindings b),'links',(SELECT jsonb_agg(to_jsonb(l) ORDER BY audit_id,logical_key) FROM audit_artifact_links l))::text`).Scan(&value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	before := snapshot()
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available); err != nil {
		t.Fatal(err)
	}
	if got := snapshot(); got != before {
		t.Fatal("approval removal changed exact ordinary Artifact data")
	}
	for _, table := range []string{"project_findings", "project_finding_sources", "finding_proposal_receipts"} {
		var count int
		if err := pool.QueryRow(ctx, "SELECT count(*) FROM "+table).Scan(&count); err != nil || count != 0 {
			t.Fatalf("historical facts promoted into %s: %d %v", table, count, err)
		}
	}

	var policy string
	var revision int
	if err := pool.QueryRow(ctx, `SELECT active_checks,revision FROM projects WHERE project_id='scope-project'`).Scan(&policy, &revision); err != nil || policy != "prohibited" || revision != 3 {
		t.Fatalf("policy/revision: %s/%d %v", policy, revision, err)
	}
	for _, table := range []string{"audit_review_requests", "audit_review_decisions", "audit_live_item_reviews"} {
		var absent bool
		if err := pool.QueryRow(ctx, `SELECT to_regclass($1) IS NULL`, table).Scan(&absent); err != nil || !absent {
			t.Fatalf("obsolete relation %s survived: %v", table, err)
		}
	}
	if _, err := pool.Exec(ctx, `UPDATE projects SET active_checks='approval-required',revision=revision+1,updated_at=clock_timestamp() WHERE project_id='scope-project'`); err == nil {
		t.Fatal("removed Project policy accepted")
	}
	if _, err := pool.Exec(ctx, `UPDATE audits SET state='waiting_review',started_at=clock_timestamp() WHERE audit_id='scope-audit'`); err == nil {
		t.Fatal("removed Audit state accepted")
	}
	if err := VerifyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresRemoveAuditApprovalsRefusesUnarchivedDecisionsAtomically(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, dsn)
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:6]); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, auditScopeMigrationFixtureSQL); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO audit_review_requests(request_id,audit_id,subject_kind,subject_id,kind,subject_revision,subject_digest,requested_actions,state,idempotency_key,request_digest) VALUES('unarchived','scope-audit','audit-item-action','old-item','active-check-approval',1,'sha256:'||repeat('a',64),'["approve"]','decided','unarchived','sha256:'||repeat('a',64))`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO audit_review_decisions(decision_id,request_id,audit_id,actor_id,action,rationale,subject_revision,subject_digest,idempotency_key,request_digest) VALUES('unarchived-decision','unarchived','scope-audit','owner','approve','Retain original consent metadata',1,'sha256:'||repeat('a',64),'unarchived-decision','sha256:'||repeat('a',64))`); err != nil {
		t.Fatal(err)
	}

	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available); err == nil || !strings.Contains(err.Error(), "unarchived Audit approval records") {
		t.Fatalf("unarchived decision migration: %v", err)
	}
	var requests int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM audit_review_decisions WHERE decision_id='unarchived-decision'`).Scan(&requests); err != nil || requests != 1 {
		t.Fatalf("blocked migration lost history: %d %v", requests, err)
	}
	var policy string
	if err := pool.QueryRow(ctx, `SELECT active_checks FROM projects WHERE project_id='scope-project'`).Scan(&policy); err != nil || policy != "automatic" {
		t.Fatalf("blocked migration changed Project: %s %v", policy, err)
	}
}
