package postgres

import (
	"os"
	"testing"
)

func TestProjectFindingMigrationArchivesAuditsAndDropsOldFindings(t *testing.T) {
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
	if len(available) < 3 || available[2].name != "000003_project_findings.sql" {
		t.Fatal("expected Project Finding migration after the current baseline")
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:2]); err != nil {
		t.Fatal(err)
	}
	fixture, err := os.ReadFile("testdata/project-findings-migration.sql")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, string(fixture)); err != nil {
		t.Fatal(err)
	}
	var before string
	if err := pool.QueryRow(ctx, "SELECT jsonb_agg(to_jsonb(hold) ORDER BY receipt_id)::text FROM finding_proposal_audit_holds AS hold").Scan(&before); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	for _, check := range []struct {
		query string
		want  int
	}{
		{"SELECT count(*) FROM project_findings", 0},
        {"SELECT count(*) FROM audits",0},
        {"SELECT count(*) FROM information_schema.tables WHERE table_schema=current_schema() AND table_name IN ('audit_findings','audit_finding_assessments','audit_finding_contributions')",0},
        {"SELECT count(*) FROM artifact_bindings WHERE namespace='historical-audits' AND name LIKE '%.snapshot.%'",1},
		{"SELECT count(*) FROM project_finding_sources", 0},
		{"SELECT count(*) FROM finding_proposal_receipts", 0},
		{"SELECT count(*) FROM artifact_bindings WHERE namespace='historical-audits' AND name LIKE '%.review_decisions.%'", 2},
		{"SELECT count(*) FROM artifact_bindings WHERE namespace='historical-audits' AND name LIKE '%.finding_assessments.%'", 1},
		{"SELECT count(*) FROM artifact_blobs WHERE payload='frozen report snapshot'::bytea", 1},
	} {
		var got int
		if err := pool.QueryRow(ctx, check.query).Scan(&got); err != nil || got != check.want {
			t.Fatalf("%s: got %d want %d err=%v", check.query, got, check.want, err)
		}
	}
	var after string
	if err := pool.QueryRow(ctx, "SELECT jsonb_agg(fact->'original' ORDER BY fact#>>'{original,receipt_id}')::text FROM (SELECT convert_from(b.payload,'UTF8')::jsonb AS fact FROM artifact_binding_revisions r JOIN artifact_versions v USING(version_id) JOIN artifact_blobs b ON b.sha256=v.blob_sha256 WHERE r.namespace='historical-audits' AND r.name LIKE '%.finding_proposal_holds.%') AS history").Scan(&after); err != nil || before != after {
		t.Fatalf("historical exact Audit receipt selections changed: %v", err)
	}
	var confirmed, duplicate bool
	if err := pool.QueryRow(ctx, "SELECT bool_or(convert_from(b.payload,'UTF8')::jsonb#>>'{original,verdict}'='true_positive'),bool_or(convert_from(b.payload,'UTF8')::jsonb#>>'{original,verdict}'='duplicate') FROM artifact_binding_revisions r JOIN artifact_versions v USING(version_id) JOIN artifact_blobs b ON b.sha256=v.blob_sha256 WHERE r.namespace='historical-audits'").Scan(&confirmed, &duplicate); err != nil || !confirmed || !duplicate {
		t.Fatalf("original annotations lost: confirmed=%v duplicate=%v err=%v", confirmed, duplicate, err)
	}
}
