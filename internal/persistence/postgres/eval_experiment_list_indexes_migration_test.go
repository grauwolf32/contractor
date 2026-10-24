package postgres

import (
	"maps"
	"os"
	"testing"
)

// TestSharedPostgresEvalExperimentListIndexesDropOnlyTheFilterIndexes applies
// the V414-026 migration to the schema before it and compares the catalog:
// exactly the experiment list's state, dataset and control-mode page indexes
// are gone, and every other index, primary key, UNIQUE constraint and foreign
// key is unchanged, among them the created-order page indexes the list reads
// and the dataset revision foreign key's probe index.
func TestSharedPostgresEvalExperimentListIndexesDropOnlyTheFilterIndexes(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "eval_experiment_list_indexes")
	installMigrationPrefix(t, ctx, pool, previous)
	beforeIndexes := consolidationCatalog(t, ctx, pool, consolidationIndexesSQL)
	beforeConstraints := consolidationCatalog(t, ctx, pool, consolidationConstraintsSQL)
	applyMigrationsThrough(t, ctx, pool, "eval_experiment_list_indexes")
	afterIndexes := consolidationCatalog(t, ctx, pool, consolidationIndexesSQL)
	afterConstraints := consolidationCatalog(t, ctx, pool, consolidationConstraintsSQL)

	wantIndexes := maps.Clone(beforeIndexes)
	for _, name := range []string{
		"eval_experiments_state_page_idx", "eval_experiments_dataset_page_idx", "eval_experiments_control_page_idx",
	} {
		if _, present := wantIndexes[name]; !present {
			t.Fatalf("index %s is missing before the migration", name)
		}
		delete(wantIndexes, name)
	}
	for _, kept := range []string{
		"eval_experiments_owner_created_page_idx", "eval_experiments_project_created_page_idx",
		"eval_experiments_dataset_revision_idx",
	} {
		if _, present := wantIndexes[kept]; !present {
			t.Fatalf("index %s is missing before the migration", kept)
		}
	}
	if !maps.Equal(afterIndexes, wantIndexes) {
		t.Errorf("indexes after the migration differ from the intended change:\n%s", consolidationDiff(wantIndexes, afterIndexes))
	}
	if !maps.Equal(afterConstraints, beforeConstraints) {
		t.Errorf("constraints changed:\n%s", consolidationDiff(beforeConstraints, afterConstraints))
	}
}
