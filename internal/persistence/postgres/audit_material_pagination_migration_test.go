package postgres

import (
	"os"
	"testing"
)

func TestAuditMaterialPaginationPreservesAcceptedLinks(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, dsn)
	migrations, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	if len(migrations) < 13 || migrations[12].name != "000013_audit_material_pagination.sql" {
		t.Fatal("missing material pagination migration")
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:6]); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, auditScopeMigrationFixtureSQL); err != nil {
		t.Fatal(err)
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:12]); err != nil {
		t.Fatal(err)
	}
	snapshot := func() string {
		t.Helper()
		var result string
		if err := pool.QueryRow(ctx, `SELECT jsonb_agg(to_jsonb(l)-'material_ordinal' ORDER BY audit_id,logical_key)::text FROM audit_artifact_links l`).Scan(&result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	before := snapshot()
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, migrations[:13]); err != nil {
		t.Fatal(err)
	}
	if snapshot() != before {
		t.Fatal("pagination metadata changed accepted material identity or provenance")
	}
	var count, unique int
	var first, last int64
	if err := pool.QueryRow(ctx, `SELECT count(*),count(DISTINCT material_ordinal),min(material_ordinal),max(material_ordinal) FROM audit_artifact_links`).Scan(&count, &unique, &first, &last); err != nil || count == 0 || count != unique || first < 1 {
		t.Fatalf("invalid backfilled order: %d/%d/%d/%d %v", count, unique, first, last, err)
	}
	var next int64
	if err := pool.QueryRow(ctx, `INSERT INTO audit_artifact_links(audit_id,artifact_scope,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance) SELECT audit_id,artifact_scope,'pagination-copy',artifact_ref,artifact_digest,media_type,size_bytes,source_provenance FROM audit_artifact_links ORDER BY material_ordinal LIMIT 1 RETURNING material_ordinal`).Scan(&next); err != nil || next <= last {
		t.Fatalf("new material did not advance order: %d <= %d: %v", next, last, err)
	}
}
