package postgres

import (
	"os"
	"strings"
	"testing"
)

func TestPostgresLLMCostMigrationPreservesRowsAndAddsNoTables(t *testing.T) {
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
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:1]); err != nil {
		t.Fatal(err)
	}
	insert := `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest) VALUES ($1,'owner','project','Preserved',$1,$2)`
	if _, err := pool.Exec(ctx, insert, "before", "sha256:"+strings.Repeat("a", 64)); err != nil {
		t.Fatal(err)
	}
	count := func() int {
		t.Helper()
		var value int
		if err := pool.QueryRow(ctx, `SELECT count(*) FROM pg_tables WHERE schemaname=current_schema()`).Scan(&value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	tables := count()
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available); err != nil {
		t.Fatal(err)
	}
	if count() != tables {
		t.Fatal("cost migration introduced a table")
	}
	if _, err := pool.Exec(ctx, insert, "after", "sha256:"+strings.Repeat("b", 64)); err != nil {
		t.Fatal(err)
	}
	var oldUnknown, newKnown bool
	var name string
	var revision int64
	if err := pool.QueryRow(ctx, `SELECT name,revision,llm_cost IS NULL FROM projects WHERE project_id='before'`).Scan(&name, &revision, &oldUnknown); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT llm_cost='{}'::jsonb FROM projects WHERE project_id='after'`).Scan(&newKnown); err != nil {
		t.Fatal(err)
	}
	if name != "Preserved" || revision != 1 || !oldUnknown || !newKnown {
		t.Fatalf("migration changed business data or fabricated history: %q %d %t %t", name, revision, oldUnknown, newKnown)
	}
}
