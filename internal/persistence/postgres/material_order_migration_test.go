package postgres

import (
	"bytes"
	"os"
	"testing"
)

func TestPostgresMaterialOrderMigrationPreservesExistingBindings(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:4]); err != nil {
		t.Fatal(err)
	}
	for _, sql := range []string{
		`INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest) VALUES ('material-before','owner','project','Preserved','before','sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')`,
		`INSERT INTO artifact_scopes(scope_kind,scope_id) VALUES ('project','material-before')`,
		`INSERT INTO artifact_blobs(sha256,payload,size_bytes,backend) VALUES (sha256('fixture'::bytea),'fixture',7,'postgresql')`,
		`INSERT INTO artifact_versions(version_id,blob_sha256,media_type) VALUES ('v-before',sha256('fixture'::bytea),'application/vnd.oai.openapi+json')`,
		`WITH binding AS (
         INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,retained_revisions,retained_bytes)
         VALUES ('project','material-before','outputs','api','r-before',1,7) RETURNING 1)
         INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id)
         SELECT 'project','material-before','outputs','api','r-before','v-before' FROM binding`,
	} {
		if _, err := pool.Exec(ctx, sql); err != nil {
			t.Fatal(err)
		}
	}
	var before, after []byte
	if err := pool.QueryRow(ctx, `SELECT to_jsonb(b) FROM artifact_bindings b WHERE scope_id='material-before'`).Scan(&before); err != nil {
		t.Fatal(err)
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available); err != nil {
		t.Fatal(err)
	}
	var rank int
	var payload []byte
	if err := pool.QueryRow(ctx, `SELECT to_jsonb(b)-'material_rank',material_rank FROM artifact_bindings b WHERE scope_id='material-before'`).Scan(&after, &rank); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT payload FROM artifact_blobs WHERE sha256=sha256('fixture'::bytea)`).Scan(&payload); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) || rank != 1 || string(payload) != "fixture" {
		t.Fatalf("migration changed existing data: rank=%d before=%s after=%s", rank, before, after)
	}
}
