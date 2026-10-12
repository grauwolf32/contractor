//go:build integration

package artifacts_test

import (
	"context"
	"errors"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// runRevisionLimit is LIM-ART-6: binding revisions one RunScope may create
// outside its Scheduler-managed outputs Namespace.
const runRevisionLimit = 50000

// sharedInputs is how many seeded Run inputs are forks of UserScope artifacts,
// sharing their versions and blobs.
const sharedInputs = 100

// A Run fills up to its revision limit, the limit holds under concurrent
// writers, outputs stay bindable, and Run deletion then purges all 50,000
// revisions in one transaction whose statements each handle at most one purge
// batch, whose statement count grows with the batches rather than the rows,
// and which reads each revision and binding a bounded number of times. Seeded
// rows use inline PostgreSQL blobs whatever the test backend: the filesystem
// unlink of a batched purge is covered by
// TestPostgresIntegrationProjectScopePurgeCommitsBoundedBatches.
func TestPostgresRunRevisionLimitAndBoundedPurge(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	createArtifactRun(t, ctx, pool, "limit-run", true)
	seedRunRevisions(t, ctx, pool, "limit-run", runRevisionLimit-1)

	// The last free revision goes to exactly one of two concurrent writers.
	first, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Rollback(context.Background())
	second, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Rollback(context.Background())
	write := func(db postgres.DBTX, name string) (WriteResult, error) {
		run, _ := NewService(NewPostgresRepository(db)).Run("limit-run")
		return run.Write(ctx, ArtifactRef{Namespace: "scratch", Name: name}, Payload{MediaType: "text/plain", Data: []byte(name)}, nil)
	}
	last, err := write(first, "last")
	if err != nil {
		t.Fatalf("write the last free revision: %v", err)
	}
	done := make(chan error, 1)
	go func() {
		_, err := write(second, "concurrent")
		done <- err
	}()
	waitArtifactLock(t, ctx, pool, second.Conn().PgConn().PID())
	if err := first.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-done; !errors.Is(err, ErrRunRevisionLimit) {
		t.Fatalf("concurrent write beyond the limit = %v", err)
	}
	if err := second.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := write(pool, "late"); !errors.Is(err, ErrRunRevisionLimit) {
		t.Fatalf("write beyond the limit = %v", err)
	}
	var revisions, counted, bindings int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind = 'run' AND scope_id = 'limit-run'),
       (SELECT revision_count FROM artifact_scopes WHERE scope_kind = 'run' AND scope_id = 'limit-run'),
       (SELECT count(*) FROM artifact_bindings WHERE scope_kind = 'run' AND scope_id = 'limit-run')`,
	).Scan(&revisions, &counted, &bindings); err != nil {
		t.Fatal(err)
	}
	if revisions != runRevisionLimit || counted != runRevisionLimit {
		t.Fatalf("Run revisions = %d, counted %d, want %d", revisions, counted, runRevisionLimit)
	}
	// The limit is per Run: another Run still writes.
	createArtifactRun(t, ctx, pool, "other-run", true)
	other, _ := NewService(NewPostgresRepository(pool)).Run("other-run")
	if _, err := other.Write(ctx, ArtifactRef{Namespace: "scratch", Name: "free"}, Payload{MediaType: "text/plain", Data: []byte("free")}, nil); err != nil {
		t.Fatalf("write to another Run: %v", err)
	}
	// The Scheduler still binds outputs of an exhausted Run.
	if _, err := NewService(NewPostgresRepository(pool)).BindOutputExact(ctx, "limit-run", "result", last.Ref, nil); err != nil {
		t.Fatalf("bind output of an exhausted Run: %v", err)
	}
	purged, bindings := revisions+1, bindings+1 // with the outputs revision and binding

	if _, err := runstore.NewPostgresStore(pool).TransitionRun(ctx, "limit-run", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
		t.Fatal(err)
	}
	var recorded statementRecord
	if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		recorder := &recordingTx{Tx: tx, record: &recorded}
		if err := rundeletion.New(recorder).DeleteReleasedTerminalRun(ctx, "user-1", "limit-run"); err != nil {
			return err
		}
		// Rows this transaction read from the revision and binding tables,
		// before commit flushes the transaction's statistics.
		return tx.QueryRow(ctx, `
SELECT coalesce(sum(seq_tup_read + idx_tup_fetch) FILTER (WHERE relname = 'artifact_binding_revisions'), 0),
       coalesce(sum(seq_tup_read + idx_tup_fetch) FILTER (WHERE relname = 'artifact_bindings'), 0)
FROM pg_stat_xact_user_tables`).Scan(&recorded.revisionReads, &recorded.bindingReads)
	}); err != nil {
		t.Fatalf("delete the Run at its revision limit: %v", err)
	}
	if recorded.largestRows > int64(PurgeBatchSize()) {
		t.Fatalf("one purge statement handled %d rows, more than the batch of %d: %s",
			recorded.largestRows, PurgeBatchSize(), recorded.largestSQL)
	}
	batches := (purged + PurgeBatchSize() - 1) / PurgeBatchSize()
	if recorded.statements > 12*batches+64 {
		t.Fatalf("Run purge issued %d statements for %d batches", recorded.statements, batches)
	}
	// Rereading the scope for every batch would read about
	// purged*purged/(2*batch) rows, 25 times the revisions here.
	if recorded.revisionReads > int64(10*purged) || recorded.bindingReads > int64(10*bindings) {
		t.Fatalf("Run purge read %d revision rows for %d revisions and %d binding rows for %d bindings",
			recorded.revisionReads, purged, recorded.bindingReads, bindings)
	}
	var runRows, versions, blobs, sharedVersions int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM artifact_scopes WHERE scope_id = 'limit-run')
     + (SELECT count(*) FROM artifact_bindings WHERE scope_id = 'limit-run')
     + (SELECT count(*) FROM artifact_binding_revisions WHERE scope_id = 'limit-run')
     + (SELECT count(*) FROM artifact_lineage WHERE target_scope_id = 'limit-run' OR source_scope_id = 'limit-run'),
       (SELECT count(*) FROM artifact_versions),
       (SELECT count(*) FROM artifact_blobs),
       (SELECT count(*) FROM artifact_versions WHERE version_id LIKE 'shared-version-%')`,
	).Scan(&runRows, &versions, &blobs, &sharedVersions); err != nil {
		t.Fatal(err)
	}
	if runRows != 0 {
		t.Fatalf("Run purge left %d Run-owned Artifact rows", runRows)
	}
	// What remains is the UserScope content the Run's inputs shared and the
	// one artifact of the other Run.
	if sharedVersions != sharedInputs || versions != sharedInputs+1 || blobs != sharedInputs+1 {
		t.Fatalf("Run purge kept %d versions (%d shared) and %d blobs, want %d shared plus one each",
			versions, sharedVersions, blobs, sharedInputs)
	}
}

// seedRunRevisions creates count binding revisions in a RunScope in a few
// set-based statements, through the revision limit trigger like any other
// write. The first sharedInputs are inputs forked from UserScope artifacts,
// with their lineage and versions; the others are scratch revisions with
// their own versions and inline blobs, about three per binding, so binding
// groups straddle purge batches.
func seedRunRevisions(t *testing.T, ctx context.Context, pool postgres.DBTX, runID string, count int) {
	t.Helper()
	shared := []any{sharedInputs}
	run := []any{runID, sharedInputs}
	scratch := []any{runID, sharedInputs, count}
	for _, statement := range []struct {
		sql       string
		arguments []any
	}{{`
INSERT INTO artifact_scopes (scope_kind, scope_id) VALUES ('user', 'user-1'), ('run', $1)`, []any{runID}}, {`
INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend)
SELECT sha256(convert_to('shared-' || i, 'UTF8')), convert_to('shared-' || i, 'UTF8'),
       octet_length(convert_to('shared-' || i, 'UTF8')), 'postgresql'
FROM generate_series(1, $1::int) AS i`, shared}, {`
INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
SELECT 'shared-version-' || i, sha256(convert_to('shared-' || i, 'UTF8')), 'text/plain'
FROM generate_series(1, $1::int) AS i`, shared}, {`
INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
SELECT 'user', 'user-1', 'sources', 'source-' || i, 'rev-' || i, 'shared-version-' || i
FROM generate_series(1, $1::int) AS i`, shared}, {`
INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision)
SELECT 'user', 'user-1', 'sources', 'source-' || i, 'rev-' || i
FROM generate_series(1, $1::int) AS i`, shared}, {`
INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
SELECT 'run', $1, 'inputs', 'source-' || i, 'fork-' || i, 'shared-version-' || i
FROM generate_series(1, $2::int) AS i`, run}, {`
INSERT INTO artifact_lineage (
    target_scope_kind, target_scope_id, target_namespace, target_name, target_revision,
    source_scope_kind, source_scope_id, source_namespace, source_name, source_revision, lineage_kind)
SELECT 'run', $1, 'inputs', 'source-' || i, 'fork-' || i,
       'user', 'user-1', 'sources', 'source-' || i, 'rev-' || i, 'input_fork'
FROM generate_series(1, $2::int) AS i`, run}, {`
INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend)
SELECT sha256(convert_to($1 || '-' || i, 'UTF8')), convert_to($1 || '-' || i, 'UTF8'),
       octet_length(convert_to($1 || '-' || i, 'UTF8')), 'postgresql'
FROM generate_series($2::int + 1, $3::int) AS i`, scratch}, {`
INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
SELECT $1 || '-version-' || i, sha256(convert_to($1 || '-' || i, 'UTF8')), 'text/plain'
FROM generate_series($2::int + 1, $3::int) AS i`, scratch}, {`
INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
SELECT 'run', $1, 'scratch', 'item-' || (i % 16667), 'rev-' || i, $1 || '-version-' || i
FROM generate_series($2::int + 1, $3::int) AS i`, scratch}, {`
INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision)
SELECT DISTINCT ON (namespace, name) scope_kind, scope_id, namespace, name, revision
FROM artifact_binding_revisions
WHERE scope_kind = 'run' AND scope_id = $1
ORDER BY namespace, name, revision DESC`, []any{runID}}} {
		if _, err := pool.Exec(ctx, statement.sql, statement.arguments...); err != nil {
			t.Fatalf("seed Run revisions: %v", err)
		}
	}
}

// statementRecord summarizes the statements of one purge transaction.
type statementRecord struct {
	statements    int
	largestRows   int64
	largestSQL    string
	revisionReads int64
	bindingReads  int64
}

// recordingTx records the number of statements and the statement that
// affected or returned the most rows.
type recordingTx struct {
	pgx.Tx
	record *statementRecord
}

func (tx *recordingTx) observe(tag pgconn.CommandTag, sql string) {
	tx.record.statements++
	if rows := tag.RowsAffected(); rows > tx.record.largestRows {
		tx.record.largestRows, tx.record.largestSQL = rows, sql
	}
}

func (tx *recordingTx) Exec(ctx context.Context, sql string, arguments ...any) (pgconn.CommandTag, error) {
	tag, err := tx.Tx.Exec(ctx, sql, arguments...)
	tx.observe(tag, sql)
	return tag, err
}

func (tx *recordingTx) Query(ctx context.Context, sql string, arguments ...any) (pgx.Rows, error) {
	rows, err := tx.Tx.Query(ctx, sql, arguments...)
	if err != nil {
		tx.observe(pgconn.CommandTag{}, sql)
		return rows, err
	}
	return &recordingRows{Rows: rows, observe: func(tag pgconn.CommandTag) { tx.observe(tag, sql) }}, nil
}

func (tx *recordingTx) QueryRow(ctx context.Context, sql string, arguments ...any) pgx.Row {
	rows, err := tx.Query(ctx, sql, arguments...)
	return recordedRow{rows: rows, err: err}
}

// recordingRows reports the command tag once the rows are closed or drained.
type recordingRows struct {
	pgx.Rows
	observe func(pgconn.CommandTag)
	closed  bool
}

func (rows *recordingRows) Close() {
	rows.Rows.Close()
	if !rows.closed {
		rows.closed = true
		rows.observe(rows.Rows.CommandTag())
	}
}

func (rows *recordingRows) Next() bool {
	if rows.Rows.Next() {
		return true
	}
	rows.Close()
	return false
}

type recordedRow struct {
	rows pgx.Rows
	err  error
}

func (row recordedRow) Scan(dest ...any) error {
	if row.err != nil {
		return row.err
	}
	defer row.rows.Close()
	if !row.rows.Next() {
		if err := row.rows.Err(); err != nil {
			return err
		}
		return pgx.ErrNoRows
	}
	return row.rows.Scan(dest...)
}
