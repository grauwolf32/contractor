package postgres

import (
	"os"
	"testing"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
)

// TestSharedPostgresFindingCorrelationFenceMigrationKeepsEveryRevision applies
// the V414-030 migration to Projects whose revisions advanced, with and
// without an Audit (FND-R177). Every Project keeps its exact revision in its
// own fence row, projects loses the column, its trigger and the functions
// that wrote it, and the fence row keeps the old guarantees: it steps by one
// per call of the advance function, rejects a decrease or a skip, stays while
// its Project exists and goes with it.
func TestSharedPostgresFindingCorrelationFenceMigrationKeepsEveryRevision(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "finding_correlation_fences")
	installMigrationPrefix(t, ctx, pool, previous)
	want := map[string]int64{"fence-audited": 5, "fence-run-only": 2, "fence-untouched": 1}
	for project, revision := range want {
		if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
 VALUES($1,'owner','project','Keep '||$1,$1,'sha256:'||repeat('a',64))`, project); err != nil {
			t.Fatal(err)
		}
		// The old column only stepped by one.
		for step := int64(1); step < revision; step++ {
			if _, err := pool.Exec(ctx, `UPDATE projects SET finding_correlation_revision=finding_correlation_revision+1
 WHERE project_id=$1`, project); err != nil {
				t.Fatal(err)
			}
		}
	}
	if _, err := pool.Exec(ctx, `INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,
 profile_snapshot,input_selection,max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,
 max_item_run_attempts,max_evidence_bytes)
 VALUES('fence-audit','owner','fence-audited','profile','1','sha256:'||repeat('a',64),'{}','{}',1,1,10,10,10,1,1024)`); err != nil {
		t.Fatal(err)
	}
	applyMigrationsThrough(t, ctx, pool, "finding_correlation_fences")

	readFences := func() map[string]int64 {
		t.Helper()
		rows, err := pool.Query(ctx, `SELECT project_id, revision FROM finding_correlation_fences`)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()
		fences := map[string]int64{}
		for rows.Next() {
			var project string
			var revision int64
			if err := rows.Scan(&project, &revision); err != nil {
				t.Fatal(err)
			}
			fences[project] = revision
		}
		if err := rows.Err(); err != nil {
			t.Fatal(err)
		}
		return fences
	}
	if fences := readFences(); len(fences) != len(want) || fences["fence-audited"] != 5 ||
		fences["fence-run-only"] != 2 || fences["fence-untouched"] != 1 {
		t.Fatalf("migrated fences = %v, want %v", fences, want)
	}
	var unfencedAudits int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM audits AS audit
 WHERE NOT EXISTS (SELECT 1 FROM finding_correlation_fences AS fence WHERE fence.project_id = audit.project_id)`,
	).Scan(&unfencedAudits); err != nil || unfencedAudits != 0 {
		t.Fatalf("Audits without a fence row = %d, %v", unfencedAudits, err)
	}
	var name string
	if err := pool.QueryRow(ctx, `SELECT name FROM projects WHERE project_id='fence-audited'`).Scan(&name); err != nil ||
		name != "Keep fence-audited" {
		t.Fatalf("Project after the migration = %q, %v", name, err)
	}
	var leftovers int
	if err := pool.QueryRow(ctx, `SELECT
 (SELECT count(*) FROM information_schema.columns
   WHERE table_schema = current_schema() AND table_name = 'projects' AND column_name = 'finding_correlation_revision')
 + (SELECT count(*) FROM pg_trigger WHERE tgrelid = 'projects'::regclass AND tgname = 'projects_protect_correlation_revision')
 + (SELECT count(*) FROM pg_proc WHERE pronamespace = current_schema()::regnamespace
     AND proname IN ('contractor_protect_finding_correlation_revision', 'contractor_lock_finding_correlation_project'))`,
	).Scan(&leftovers); err != nil || leftovers != 0 {
		t.Fatalf("projects keeps %d correlation column, trigger or function, %v", leftovers, err)
	}
	// No function of the schema writes projects or locks it exclusively any
	// more; only projectstore's Go SQL does (PER-R23).
	var writers []string
	writersSQL := `SELECT coalesce(array_agg(proname ORDER BY proname), '{}') FROM pg_proc
 WHERE pronamespace = current_schema()::regnamespace
   AND ` + explaintest.RoutineBodySQL("") + ` ~* '(UPDATE|INSERT\s+INTO|DELETE\s+FROM)\s+projects\M|FROM\s+projects\M[^;]*FOR\s+(NO\s+KEY\s+)?UPDATE'`
	if err := pool.QueryRow(ctx, writersSQL).Scan(&writers); err != nil || len(writers) != 0 {
		t.Fatalf("functions that write or exclusively lock projects = %v, %v", writers, err)
	}

	// Prove this check can see a SQL-standard writer, then remove the
	// test-only function before continuing the migration's behavior checks.
	if _, err := pool.Exec(ctx, `CREATE FUNCTION contractor_test_project_writer() RETURNS void LANGUAGE sql
BEGIN ATOMIC UPDATE projects SET name = name WHERE false; END`); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, writersSQL).Scan(&writers); err != nil || len(writers) != 1 || writers[0] != "contractor_test_project_writer" {
		t.Fatalf("SQL-standard Project writer = %v, %v", writers, err)
	}
	if _, err := pool.Exec(ctx, `DROP FUNCTION contractor_test_project_writer()`); err != nil {
		t.Fatal(err)
	}

	// The triggers step each affected Project once per statement, whatever
	// the number of its rows.
	if _, err := pool.Exec(ctx, `SELECT contractor_advance_finding_correlation(
 ARRAY['fence-audited','fence-run-only','fence-audited'])`); err != nil {
		t.Fatal(err)
	}
	if fences := readFences(); fences["fence-audited"] != 6 || fences["fence-run-only"] != 3 || fences["fence-untouched"] != 1 {
		t.Fatalf("fences after one advance = %v", fences)
	}
	for _, change := range []string{"revision-1", "revision+2", "0"} {
		if _, err := pool.Exec(ctx, `UPDATE finding_correlation_fences SET revision=`+change+
			` WHERE project_id='fence-audited'`); Classify(err) != ClassRejected {
			t.Fatalf("revision change %s = %v", change, err)
		}
	}
	if _, err := pool.Exec(ctx, `DELETE FROM finding_correlation_fences WHERE project_id='fence-untouched'`); Classify(err) != ClassRejected {
		t.Fatalf("fence deletion while its Project exists = %v", err)
	}

	// A Project created after the migration has no fence row until its first
	// Audit, and the lock of its missing fence is an invariant failure.
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
 VALUES('fence-new','owner','project','New','fence-new','sha256:'||repeat('a',64))`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `SELECT contractor_lock_finding_correlation_fence('fence-new')`); Classify(err) != ClassInternal {
		t.Fatalf("lock of a missing fence = %v", err)
	}

	// The fence row goes with its Project.
	if _, err := pool.Exec(ctx, `DELETE FROM projects WHERE project_id='fence-untouched'`); err != nil {
		t.Fatal(err)
	}
	if fences := readFences(); len(fences) != 2 || fences["fence-untouched"] != 0 {
		t.Fatalf("fences after the Project deletion = %v", fences)
	}
}
