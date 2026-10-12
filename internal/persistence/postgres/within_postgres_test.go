package postgres

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// TestPostgresWithinRetriesOnAPoolAndJoinsACallerTransaction pins the one
// pool-or-transaction rule: on a pool Within replays a definite transaction
// conflict in a fresh transaction; on a caller's transaction it runs once
// inside it, sees the caller's uncommitted writes and leaves the conflict to
// the caller.
func TestPostgresWithinRetriesOnAPoolAndJoinsACallerTransaction(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, url)
	if _, err := pool.Exec(ctx, `CREATE TABLE within_marks (id integer PRIMARY KEY)`); err != nil {
		t.Fatal(err)
	}

	attempts := 0
	err := Within(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		attempts++
		if _, err := tx.Exec(ctx, `INSERT INTO within_marks VALUES (1)`); err != nil {
			return err
		}
		if attempts == 1 {
			return &pgconn.PgError{Code: SQLStateSerializationFailure}
		}
		return nil
	})
	var rows int
	if countErr := pool.QueryRow(ctx, `SELECT count(*) FROM within_marks`).Scan(&rows); err != nil || countErr != nil ||
		attempts != 2 || rows != 1 {
		t.Fatalf("pool Within = %v after %d attempts with %d rows (%v), want one replay", err, attempts, rows, countErr)
	}

	tx, err := BeginTx(ctx, pool, pgx.TxOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if _, err := tx.Exec(ctx, `INSERT INTO within_marks VALUES (2)`); err != nil {
		t.Fatal(err)
	}
	calls := 0
	err = Within(ctx, tx, pgx.TxOptions{}, func(inner pgx.Tx) error {
		calls++
		var joined bool
		if err := inner.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM within_marks WHERE id = 2)`).Scan(&joined); err != nil {
			return err
		}
		if !joined {
			return errors.New("Within did not join the caller's transaction")
		}
		return &pgconn.PgError{Code: SQLStateSerializationFailure}
	})
	if calls != 1 || !IsTransactionConflict(err) {
		t.Fatalf("transaction Within = %v after %d calls, want the caller's conflict after one call", err, calls)
	}

	ran := false
	err = Within(ctx, struct{ DBTX }{pool}, pgx.TxOptions{}, func(pgx.Tx) error {
		ran = true
		return nil
	})
	if err == nil || ran {
		t.Fatalf("Within on neither a pool nor a transaction = %v (ran %t), want a refusal", err, ran)
	}
}

// TestPostgresBeginTxAppliesTheOperationBudget checks that a transaction a
// caller begins and commits itself runs under the operation budget selected
// on its context, as an InTx transaction does.
func TestPostgresBeginTxAppliesTheOperationBudget(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, budgetTestDatabaseURL(t))
	budgets := Budgets{
		AcquireTimeout: time.Second, QueryTimeout: 9 * time.Second, StatementTimeout: 7 * time.Second,
		LockTimeout: 1500 * time.Millisecond, IdleTransactionTimeout: 11 * time.Second,
	}
	budgeted, err := WithOperationBudgets(ctx, budgets)
	if err != nil {
		t.Fatal(err)
	}
	tx, err := BeginTx(budgeted, pool, pgx.TxOptions{AccessMode: pgx.ReadOnly})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	var statement, lock, idle int64
	if err := tx.QueryRow(budgeted, `SELECT (extract(epoch FROM current_setting('statement_timeout')::interval)*1000)::bigint,
(extract(epoch FROM current_setting('lock_timeout')::interval)*1000)::bigint,
(extract(epoch FROM current_setting('idle_in_transaction_session_timeout')::interval)*1000)::bigint`).Scan(
		&statement, &lock, &idle); err != nil {
		t.Fatal(err)
	}
	if statement != budgets.StatementTimeout.Milliseconds() || lock != budgets.LockTimeout.Milliseconds() ||
		idle != budgets.IdleTransactionTimeout.Milliseconds() {
		t.Fatalf("BeginTx settings = %d/%d/%d ms, want %+v", statement, lock, idle, budgets)
	}
}
