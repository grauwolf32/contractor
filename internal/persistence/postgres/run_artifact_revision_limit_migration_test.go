package postgres

import (
	"context"
	"fmt"
	"maps"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// runRevisionLimit is LIM-ART-6: binding revisions one RunScope may create
// outside its Scheduler-managed outputs Namespace.
const runRevisionLimit = 50000

// artifactScopeKey names one artifact_scopes row.
type artifactScopeKey struct{ kind, id string }

// Migration 000008 (run_artifact_revision_limit) adds the revision counter
// of each RunScope and backfills it from the revisions an older Server
// stored: every revision of a RunScope outside its outputs Namespace
// (LIM-ART-6, ART-R118). The upgraded schema starts at version 7 with empty,
// outputs-only and populated RunScopes, one at the limit minus one, and with
// User and Project scopes, one of which shares a RunScope's id. Afterwards
// every counter equals that quota definition, computed here from the seeded
// rows, later inserts keep counting and stop at the limit, and the purge
// removes the counter with its scope. The test fails if the backfill is
// omitted, counts outputs revisions or counts another scope kind.
func TestPostgresRunArtifactRevisionLimitMigrationBackfillsRunScopes(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "run_artifact_revision_limit")
	installMigrationPrefix(t, ctx, pool, previous)

	mixed := artifactScopeKey{"run", "run-mixed"}
	empty := artifactScopeKey{"run", "run-empty"}
	outputsOnly := artifactScopeKey{"run", "run-outputs"}
	nearLimit := artifactScopeKey{"run", "run-near-limit"}
	sameID := artifactScopeKey{"user", "run-mixed"}
	project := artifactScopeKey{"project", "project-a"}
	seeds := []struct {
		scope     artifactScopeKey
		namespace string
		count     int
	}{
		{mixed, "inputs", 5}, {mixed, "scratch", 7}, {mixed, "memory", 4}, {mixed, "outputs", 6},
		{outputsOnly, "outputs", 4},
		{nearLimit, "inputs", 100}, {nearLimit, "scratch", runRevisionLimit - 101}, {nearLimit, "outputs", 3},
		{sameID, "scratch", 9},
		{project, "sources", 3}, {project, "outputs", 2},
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
VALUES ('project-a', 'user', 'project', 'Project', 'project-a', 'sha256:' || repeat('a', 64))`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version, workflow_schema_version, workflow_snapshot,
    parameters, runtime_labels, runtime_config_snapshot, state, state_reason_code, started_at
) SELECT run_id, 'user', 'workflow', '1', 'v1', '{}', '{}', '{}', $1::jsonb, 'running', 'test', clock_timestamp()
    FROM unnest(ARRAY['run-mixed', 'run-empty', 'run-outputs', 'run-near-limit']) AS run_id`, builtInRunRuntimeSnapshot); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO artifact_scopes (scope_kind, scope_id)
VALUES ('run', 'run-mixed'), ('run', 'run-empty'), ('run', 'run-outputs'), ('run', 'run-near-limit'),
       ('user', 'run-mixed'), ('project', 'project-a');
INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend)
VALUES (sha256('payload'::bytea), 'payload'::bytea, 7, 'postgresql');
INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
VALUES ('version', sha256('payload'::bytea), 'text/plain')`); err != nil {
		t.Fatal(err)
	}
	// Each binding holds about three revisions, as a Run's writes leave them.
	insertRevisions := func(db DBTX, scope artifactScopeKey, namespace, prefix string, count int) error {
		_, err := db.Exec(ctx, `
INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
SELECT $1, $2, $3, 'item-' || (i / 3), $4 || '-' || i, 'version'
  FROM generate_series(1, $5::int) AS i`, scope.kind, scope.id, namespace, prefix, count)
		return err
	}
	want := map[artifactScopeKey]int64{mixed: 0, empty: 0, outputsOnly: 0, nearLimit: 0, sameID: 0, project: 0}
	for _, seed := range seeds {
		if err := insertRevisions(pool, seed.scope, seed.namespace, "seed", seed.count); err != nil {
			t.Fatalf("seed %d %s revisions of %v: %v", seed.count, seed.namespace, seed.scope, err)
		}
		// The quota: revisions of a RunScope outside its outputs Namespace.
		if seed.scope.kind == "run" && seed.namespace != "outputs" {
			want[seed.scope] += int64(seed.count)
		}
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision)
SELECT DISTINCT ON (namespace, name) scope_kind, scope_id, namespace, name, revision
  FROM artifact_binding_revisions
 WHERE scope_kind = 'run' AND scope_id = 'run-mixed'
 ORDER BY namespace, name, created_at DESC, revision DESC`); err != nil {
		t.Fatal(err)
	}

	result, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	if err != nil {
		t.Fatalf("upgrade the populated schema: %v", err)
	}
	if len(result.AppliedVersions) == 0 || result.AppliedVersions[0] != previous+1 {
		t.Fatalf("applied versions %v, want the upgrade to start at %d", result.AppliedVersions, previous+1)
	}
	counters := func() map[artifactScopeKey]int64 {
		t.Helper()
		rows, err := pool.Query(ctx, `SELECT scope_kind, scope_id, revision_count FROM artifact_scopes`)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()
		got := map[artifactScopeKey]int64{}
		for rows.Next() {
			var scope artifactScopeKey
			var counted int64
			if err := rows.Scan(&scope.kind, &scope.id, &counted); err != nil {
				t.Fatal(err)
			}
			got[scope] = counted
		}
		if err := rows.Err(); err != nil {
			t.Fatal(err)
		}
		return got
	}
	if got := counters(); !maps.Equal(got, want) {
		t.Fatalf("backfilled revision counters = %v, want %v", got, want)
	}

	// One statement that writes to several scopes counts each RunScope's new
	// revisions outside outputs and nothing else.
	if err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, write := range []struct {
			scope     artifactScopeKey
			namespace string
			count     int
		}{{mixed, "scratch", 2}, {mixed, "outputs", 1}, {empty, "scratch", 3}, {outputsOnly, "outputs", 2}, {sameID, "scratch", 2}} {
			if err := insertRevisions(tx, write.scope, write.namespace, "later", write.count); err != nil {
				return err
			}
		}
		_, err := tx.Exec(ctx, `
INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
VALUES ('run', 'run-mixed', 'memory', 'note', 'together', 'version'),
       ('run', 'run-empty', 'outputs', 'result', 'together', 'version'),
       ('run', 'run-outputs', 'scratch', 'note', 'together', 'version'),
       ('project', 'project-a', 'sources', 'note', 'together', 'version')`)
		return err
	}); err != nil {
		t.Fatalf("write to upgraded scopes: %v", err)
	}
	want[mixed] += 3
	want[empty] += 3
	want[outputsOnly]++
	if got := counters(); !maps.Equal(got, want) {
		t.Fatalf("revision counters after later writes = %v, want %v", got, want)
	}

	// The last free revision of the RunScope at the limit minus one is
	// accepted; the next one is the limit's rejection, also when it arrives
	// with an outputs revision, and stores nothing. outputs still binds.
	if err := insertRevisions(pool, nearLimit, "scratch", "last", 1); err != nil {
		t.Fatalf("write the last free revision: %v", err)
	}
	limited := func(err error) bool {
		return Classify(err) == ClassLimitExceeded && ConstraintName(err) == "artifact_scopes_run_revision_limit"
	}
	if err := insertRevisions(pool, nearLimit, "scratch", "beyond", 1); !limited(err) {
		t.Fatalf("revision beyond the limit = %v, want the Run revision limit", err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
VALUES ('run', 'run-near-limit', 'outputs', 'result', 'beyond', 'version'),
       ('run', 'run-near-limit', 'memory', 'note', 'beyond', 'version')`); !limited(err) {
		t.Fatalf("statement beyond the limit with an outputs revision = %v, want the Run revision limit", err)
	}
	if err := insertRevisions(pool, nearLimit, "outputs", "exhausted", 2); err != nil {
		t.Fatalf("bind outputs of an exhausted Run: %v", err)
	}
	want[nearLimit] = runRevisionLimit
	var stored, beyond int64
	if err := pool.QueryRow(ctx, `
SELECT count(*) FILTER (WHERE namespace <> 'outputs'), count(*) FILTER (WHERE revision = 'beyond' OR revision LIKE 'beyond-%')
  FROM artifact_binding_revisions WHERE scope_kind = 'run' AND scope_id = 'run-near-limit'`).Scan(&stored, &beyond); err != nil {
		t.Fatal(err)
	}
	if got := counters(); !maps.Equal(got, want) || stored != runRevisionLimit || beyond != 0 {
		t.Fatalf("at the limit: counters %v, want %v; %d ordinary revisions stored, %d rejected ones",
			got, want, stored, beyond)
	}

	// The lifecycle purge of a Run deletes its revisions and then its scope
	// row in one transaction. Revisions are never given back one by one, so
	// the counter stays as it is until it goes with the scope row, its only
	// copy; the User scope with the same id keeps its rows.
	if err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `
SELECT 1 FROM artifact_scopes WHERE scope_kind = 'run' AND scope_id = 'run-mixed' FOR UPDATE;
SELECT set_config('contractor.lifecycle_purge', 'run', true);
DELETE FROM artifact_bindings WHERE scope_kind = 'run' AND scope_id = 'run-mixed';
DELETE FROM artifact_binding_revisions WHERE scope_kind = 'run' AND scope_id = 'run-mixed'`); err != nil {
			return err
		}
		var counted int64
		if err := tx.QueryRow(ctx, `
SELECT revision_count FROM artifact_scopes WHERE scope_kind = 'run' AND scope_id = 'run-mixed'`).Scan(&counted); err != nil {
			return err
		}
		if counted != want[mixed] {
			return fmt.Errorf("revision counter after the revisions were purged = %d, want %d", counted, want[mixed])
		}
		_, err := tx.Exec(ctx, `DELETE FROM artifact_scopes WHERE scope_kind = 'run' AND scope_id = 'run-mixed'`)
		return err
	}); err != nil {
		t.Fatalf("purge the upgraded Run scope: %v", err)
	}
	delete(want, mixed)
	var remaining, sameIDRevisions int
	if err := pool.QueryRow(ctx, `
SELECT count(*) FILTER (WHERE scope_kind = 'run'), count(*) FILTER (WHERE scope_kind = 'user')
  FROM artifact_binding_revisions WHERE scope_id = 'run-mixed'`).Scan(&remaining, &sameIDRevisions); err != nil {
		t.Fatal(err)
	}
	if got := counters(); !maps.Equal(got, want) || remaining != 0 || sameIDRevisions != 11 {
		t.Fatalf("after the purge: counters %v, want %v; %d purged revisions left, %d User revisions",
			got, want, remaining, sameIDRevisions)
	}
}
