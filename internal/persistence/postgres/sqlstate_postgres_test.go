package postgres

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// TestClassifyPostgreSQLConditionsOnTheMigratedSchema proves PER-R13 against
// the real server: what PostgreSQL and the schema's triggers actually report
// lands in the class that callers rely on.
func TestClassifyPostgreSQLConditionsOnTheMigratedSchema(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, other := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	const insertProject = `INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
VALUES ($1, 'owner', $2, 'Project', $1 || '-key', 'sha256:' || repeat('a', 64))`
	if _, err := pool.Exec(ctx, insertProject, "classified", "project"); err != nil {
		t.Fatal(err)
	}
	expect := func(name string, want Class, err error) {
		t.Helper()
		if got := Classify(err); got != want {
			t.Errorf("%s: Classify(%v) = %d, want %d", name, err, got, want)
		}
	}

	_, err := pool.Exec(ctx, insertProject, "classified", "project")
	expect("unique violation", ClassDuplicate, err)
	_, err = pool.Exec(ctx, insertProject, "unknown-kind", "bogus")
	expect("named CHECK constraint", ClassInvalid, err)
	_, err = pool.Exec(ctx, `INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
VALUES ('unnamed', 'owner', 'project', NULL, 'unnamed-key', 'sha256:' || repeat('a', 64))`)
	expect("not-null violation", ClassInvalid, err)
	_, err = pool.Exec(ctx, `SELECT 'not a number'::bigint`)
	expect("data exception", ClassInvalid, err)
	_, err = pool.Exec(ctx, `UPDATE projects SET kind = 'evaluation' WHERE project_id = 'classified'`)
	expect("trigger business rejection", ClassRejected, err)
	_, err = pool.Exec(ctx, `INSERT INTO artifact_scopes (scope_kind, scope_id) VALUES ('project', 'missing')`)
	expect("foreign-key violation", ClassReferenceViolation, err)
	_, err = pool.Exec(ctx, `SELECT 1 FROM classified_missing_table`)
	expect("undefined table", ClassInternal, err)

	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, `SET LOCAL statement_timeout = '20ms'; SELECT pg_sleep(1)`)
		return err
	})
	expect("statement timeout", ClassTimeout, err)

	locked, err := BeginTx(ctx, pool, pgx.TxOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := locked.Exec(ctx, `SELECT 1 FROM projects WHERE project_id = 'classified' FOR UPDATE`); err != nil {
		t.Fatal(err)
	}
	err = InTx(ctx, other, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, `SET LOCAL lock_timeout = '20ms'; SELECT 1 FROM projects WHERE project_id = 'classified' FOR UPDATE`)
		return err
	})
	expect("lock timeout", ClassTimeout, err)
	if err := locked.Rollback(ctx); err != nil {
		t.Fatal(err)
	}

	snapshot, err := BeginTx(ctx, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = snapshot.Rollback(context.WithoutCancel(ctx)) }()
	const rename = `UPDATE projects SET name = $1, revision = revision + 1, updated_at = clock_timestamp()
WHERE project_id = 'classified'`
	if _, err := snapshot.Exec(ctx, `SELECT revision FROM projects WHERE project_id = 'classified'`); err != nil {
		t.Fatal(err)
	}
	if _, err := other.Exec(ctx, rename, "Concurrent"); err != nil {
		t.Fatal(err)
	}
	_, err = snapshot.Exec(ctx, rename, "Stale")
	expect("serialization failure", ClassTransactionConflict, err)
	if !IsTransactionConflict(err) || !IsTransientFailure(err) {
		t.Fatalf("serialization failure is not a retryable transaction conflict: %v", err)
	}
	if err := snapshot.Rollback(ctx); err != nil && !errors.Is(err, pgx.ErrTxClosed) {
		t.Fatal(err)
	}

	if _, err := pool.Exec(ctx, `UPDATE projects
SET lifecycle_state = 'deleting', deletion_phase = 'cancelling', deletion_requested_at = clock_timestamp(),
    revision = revision + 1, updated_at = clock_timestamp()
WHERE project_id = 'classified'`); err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `INSERT INTO artifact_scopes (scope_kind, scope_id) VALUES ('project', 'classified')`)
	expect("Project deletion fence", ClassDeleting, err)
}
