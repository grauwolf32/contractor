package evalstore

import (
	"testing"
)

// Bulk experiment writes bump each affected collection once per statement.
// Per-row upserts rewrote the same two rows 2N times in one transaction.
func TestPostgresBulkExperimentWritesBumpCollectionsOncePerStatement(t *testing.T) {
	pool := testPool(t)
	scope := setupProject(t, pool, "bulk-owner", "bulk-project")
	ctx := t.Context()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)
	writes := func() (inserted, updated int64) {
		t.Helper()
		if err := tx.QueryRow(ctx, `
SELECT n_tup_ins, n_tup_upd
FROM pg_stat_xact_user_tables
WHERE relid = 'eval_collections'::regclass
`).Scan(&inserted, &updated); err != nil {
			t.Fatal(err)
		}
		return inserted, updated
	}
	revisions := func() (owner, project int64) {
		t.Helper()
		if err := tx.QueryRow(ctx, `
SELECT max(revision) FILTER (WHERE project_id = ''), max(revision) FILTER (WHERE project_id = $2)
FROM eval_collections
WHERE owner_id = $1
`, scope.OwnerID, scope.ProjectID).Scan(&owner, &project); err != nil {
			t.Fatal(err)
		}
		return owner, project
	}
	const experiments = 20000
	_, err = tx.Exec(ctx, `
INSERT INTO eval_experiments
  (experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms)
SELECT 'bulk-'||n, $1, $2, 'bulk-'||n, 'external', 'bulk fixture', 'finished', 1, 1000
FROM generate_series(1,$3::int) AS n`, scope.OwnerID, scope.ProjectID, experiments)
	if err != nil {
		t.Fatal(err)
	}
	if inserted, updated := writes(); inserted != 2 || updated != 0 {
		t.Fatalf("inserting %d experiments wrote collections %d+%d times, want 2 inserts", experiments, inserted, updated)
	}
	if owner, project := revisions(); owner != 1 || project != 1 {
		t.Fatalf("collection revisions after insert = %d/%d", owner, project)
	}
	if _, err = tx.Exec(ctx, `UPDATE eval_experiments
SET name = 'renamed', revision = revision + 1, updated_at = clock_timestamp()
WHERE project_id = $1`, scope.ProjectID); err != nil {
		t.Fatal(err)
	}
	if inserted, updated := writes(); inserted != 2 || updated != 2 {
		t.Fatalf("updating %d experiments wrote collections %d+%d times, want 2 updates", experiments, inserted, updated)
	}
	if owner, project := revisions(); owner != 2 || project != 2 {
		t.Fatalf("collection revisions after update = %d/%d", owner, project)
	}
	// Neither usage-only updates nor no-op assignments invalidate a list.
	if _, err = tx.Exec(ctx, `UPDATE eval_experiments
SET observed_tokens = observed_tokens + 1, name = name,
    revision = revision + 1, updated_at = clock_timestamp()
WHERE project_id = $1`, scope.ProjectID); err != nil {
		t.Fatal(err)
	}
	if inserted, updated := writes(); inserted != 2 || updated != 2 {
		t.Fatalf("usage or no-op summary update invalidated collections: %d+%d writes", inserted, updated)
	}
	if _, err = tx.Exec(ctx, `SELECT set_config('contractor.eval_purge','on',true)`); err != nil {
		t.Fatal(err)
	}
	if _, err = tx.Exec(ctx, `DELETE FROM eval_experiments WHERE project_id = $1`, scope.ProjectID); err != nil {
		t.Fatal(err)
	}
	if inserted, updated := writes(); inserted != 2 || updated != 4 {
		t.Fatalf("deleting %d experiments wrote collections %d+%d times, want 2 more updates", experiments, inserted, updated)
	}
	// Deleting moves both cursors: a list read before it must reload.
	if owner, project := revisions(); owner != 3 || project != 3 {
		t.Fatalf("collection revisions after delete = %d/%d", owner, project)
	}
}
