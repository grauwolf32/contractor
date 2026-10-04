package postgres

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"
	"testing"
	"time"
)

func TestPostgresRunMetadataLabelsFoldMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 89)

	_, err := pool.Exec(ctx, `
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version, workflow_schema_version,
    workflow_snapshot, parameters, runtime_labels, runtime_config_snapshot, state, state_reason_code
)
SELECT id, 'owner', 'workflow', '1', 'v1', '{}', '{}', '{}', $1::jsonb, 'initializing', 'created'
FROM (VALUES ('labeled'), ('plain')) AS run(id)`, builtInRunRuntimeSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO workflow_run_metadata_labels (run_id, ordinal, label_key, label_value)
VALUES ('labeled', 1, 'debug', ''), ('labeled', 2, 'eval.id', 'eval_01')`); err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 90) {
		t.Fatalf("fold Run metadata labels = %+v, %v", result, err)
	}
	var labeled, plain []byte
	var tableGone bool
	if err := pool.QueryRow(ctx, `
SELECT labeled.metadata_labels, plain.metadata_labels, to_regclass('workflow_run_metadata_labels') IS NULL
FROM workflow_runs AS labeled, workflow_runs AS plain
WHERE labeled.run_id = 'labeled' AND plain.run_id = 'plain'`).Scan(&labeled, &plain, &tableGone); err != nil {
		t.Fatal(err)
	}
	var labeledMap, plainMap map[string]string
	if json.Unmarshal(labeled, &labeledMap) != nil || json.Unmarshal(plain, &plainMap) != nil ||
		!reflect.DeepEqual(labeledMap, map[string]string{"debug": "", "eval.id": "eval_01"}) || len(plainMap) != 0 || !tableGone {
		t.Fatalf("folded labels = %s / %s, table dropped %t", labeled, plain, tableGone)
	}
	if _, err := pool.Exec(ctx, `UPDATE workflow_runs SET metadata_labels = '{}' WHERE run_id = 'labeled'`); err == nil {
		t.Fatal("immutable Run metadata labels changed")
	}
}
