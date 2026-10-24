package postgres

import (
	"errors"
	"os"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// Migration artifact_history_retention backfills the retained revisions and
// full encoded bytes of every stored binding: each revision is charged the
// size of its blob, also when two revisions share one version or blob, and a
// binding's counters never include another binding's revisions. Afterwards
// both counters are required.
func TestSharedPostgresArtifactHistoryAccountingMigrationBackfillsCounters(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "artifact_history_retention")
	installMigrationPrefix(t, ctx, pool, previous)
	err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, statement := range []string{
			`INSERT INTO artifact_scopes (scope_kind, scope_id) VALUES ('user', 'owner')`,
			`INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend)
SELECT sha256(payload), payload, octet_length(payload), 'postgresql'
FROM (VALUES ('\x616263'::bytea), ('\x6465666768'::bytea)) AS input(payload)`,
			`INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
VALUES ('v-small-1', sha256('\x616263'::bytea), 'text/plain'),
       ('v-small-2', sha256('\x616263'::bytea), 'text/plain'),
       ('v-large', sha256('\x6465666768'::bytea), 'text/plain')`,
			// history: three revisions, two sharing one version and all
			// three sharing blobs; single: one revision of the shared blob.
			`INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
VALUES ('user', 'owner', 'notes', 'history', 'r1', 'v-small-1'),
       ('user', 'owner', 'notes', 'history', 'r2', 'v-small-1'),
       ('user', 'owner', 'notes', 'history', 'r3', 'v-large'),
       ('user', 'owner', 'notes', 'single', 'r1', 'v-small-2')`,
			`INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision)
VALUES ('user', 'owner', 'notes', 'history', 'r3'), ('user', 'owner', 'notes', 'single', 'r1')`,
		} {
			if _, err := tx.Exec(ctx, statement); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	want := map[string][2]int64{"history": {3, 3 + 3 + 5}, "single": {1, 3}}
	rows, err := pool.Query(ctx, `SELECT name, retained_revisions, retained_bytes FROM artifact_bindings
WHERE scope_kind = 'user' AND scope_id = 'owner' ORDER BY name`)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string][2]int64{}
	for rows.Next() {
		var name string
		var counters [2]int64
		if err := rows.Scan(&name, &counters[0], &counters[1]); err != nil {
			t.Fatal(err)
		}
		got[name] = counters
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) || got["history"] != want["history"] || got["single"] != want["single"] {
		t.Fatalf("retained counters = %v, want %v", got, want)
	}
	_, err = pool.Exec(ctx, `INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision)
VALUES ('user', 'owner', 'notes', 'uncounted', 'r1')`)
	var rejection *pgconn.PgError
	if !errors.As(err, &rejection) || rejection.ColumnName != "retained_revisions" {
		t.Fatalf("binding without counters: %v, want a rejection of its missing retained_revisions", err)
	}
}
