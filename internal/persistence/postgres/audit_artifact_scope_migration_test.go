package postgres

import (
	"os"
	"testing"
)

func TestPostgresAuditArtifactScopeMigrationPreservesExactMaterial(t *testing.T) {
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
	if len(available) < 3 || available[2].name != "000003_audit_artifact_scope.sql" {
		t.Fatal("expected Audit scope migration")
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:2]); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, auditScopeMigrationFixtureSQL); err != nil {
		t.Fatal(err)
	}
	var before string
	if err := pool.QueryRow(ctx, `SELECT to_jsonb(a)::text FROM audits a WHERE audit_id='scope-audit'`).Scan(&before); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	var audit string
	if err := pool.QueryRow(ctx, `SELECT to_jsonb(a)::text FROM audits a WHERE audit_id='scope-audit'`).Scan(&audit); err != nil || audit != before {
		t.Fatalf("Audit business data changed: %v", err)
	}
	var version, payload, scope string
	if err := pool.QueryRow(ctx, `SELECT r.version_id,convert_from(b.payload,'UTF8'),l.artifact_scope
FROM audit_artifact_links l JOIN artifact_binding_revisions r ON r.scope_kind=l.artifact_scope AND r.scope_id=l.audit_id
 AND r.namespace=l.artifact_ref->>'namespace' AND r.name=l.artifact_ref->>'name' AND r.revision=l.artifact_ref->>'revision'
JOIN artifact_versions v USING(version_id) JOIN artifact_blobs b ON b.sha256=v.blob_sha256
WHERE l.audit_id='scope-audit' AND l.logical_key='prepare:execution:material'`).Scan(&version, &payload, &scope); err != nil || version != "shared-version" || payload != "exact material" || scope != "audit" {
		t.Fatalf("migrated exact bytes/version/scope: %s %s %s %v", version, payload, scope, err)
	}
	var versions, blobs int
	if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM artifact_versions),(SELECT count(*) FROM artifact_blobs)`).Scan(&versions, &blobs); err != nil || versions != 1 || blobs != 1 {
		t.Fatalf("migration duplicated content: %d/%d %v", versions, blobs, err)
	}
	if err := pool.QueryRow(ctx, `SELECT artifact_scope FROM audit_artifact_links WHERE logical_key='report/machine'`).Scan(&scope); err != nil || scope != "project" {
		t.Fatalf("unrelated report changed owner: %s %v", scope, err)
	}
	if err := VerifyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
}

const auditScopeMigrationFixtureSQL = `
INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES('scope-project','owner','project','Scope migration','create','sha256:'||repeat('a',64));
INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,profile_snapshot,input_selection,
max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes)
VALUES('scope-audit','owner','scope-project','fixture','1','sha256:'||repeat('b',64),'{}','{}',1,1,1,1,2,1,1048576);
INSERT INTO artifact_scopes(scope_kind,scope_id) VALUES('project','scope-project');
INSERT INTO artifact_blobs(sha256,payload,size_bytes)
VALUES(sha256('exact material'::bytea),'exact material'::bytea,14);
INSERT INTO artifact_versions(version_id,blob_sha256,media_type)
VALUES('shared-version',sha256('exact material'::bytea),'application/json');
INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id)
VALUES('project','scope-project','audit-fixture','material','exact-revision','shared-version');
INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes)
VALUES('project','scope-project','audit-fixture','material','exact-revision',true,1,14);
INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance)
VALUES('scope-audit','prepare:execution:material','{"namespace":"audit-fixture","name":"material","revision":"exact-revision"}',
'sha256:'||encode(sha256('exact material'::bytea),'hex'),'application/json',14,'{"schema":"contractor.audit.preparation-output.v1"}'),
('scope-audit','report/machine','{"namespace":"audit-fixture","name":"material","revision":"exact-revision"}',
'sha256:'||encode(sha256('exact material'::bytea),'hex'),'application/json',14,'{"schema":"report"}');
`
