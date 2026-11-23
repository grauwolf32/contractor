package postgres

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// TestPostgresSavepointQueryRowKeepsTheCallerTransactionUsable pins what a
// recovering store relies on: in a caller's transaction a failed statement
// rolls back only its own writes and the transaction continues; a statement
// that returns no rows keeps its writes, as it would on a pool.
func TestPostgresSavepointQueryRowKeepsTheCallerTransactionUsable(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, url)
	if _, err := pool.Exec(ctx, `CREATE TABLE savepoint_marks (id integer PRIMARY KEY)`); err != nil {
		t.Fatal(err)
	}
	tx, err := BeginTx(ctx, pool, pgx.TxOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if _, err := tx.Exec(ctx, `INSERT INTO savepoint_marks VALUES (1)`); err != nil {
		t.Fatal(err)
	}
	count := func(db DBTX) int {
		t.Helper()
		var rows int
		if err := db.QueryRow(ctx, `SELECT count(*) FROM savepoint_marks`).Scan(&rows); err != nil {
			t.Fatalf("the transaction is unusable: %v", err)
		}
		return rows
	}

	var id int
	err = SavepointQueryRow(ctx, tx, `WITH added AS (INSERT INTO savepoint_marks VALUES (2))
INSERT INTO savepoint_marks VALUES (1) RETURNING id`).Scan(&id)
	if Classify(err) != ClassDuplicate || count(tx) != 1 {
		t.Fatalf("failed statement = %v with %d rows, want a duplicate that leaves only the earlier row", err, count(tx))
	}
	err = SavepointQueryRow(ctx, tx, `WITH added AS (INSERT INTO savepoint_marks VALUES (3) RETURNING id)
SELECT id FROM added WHERE id < 0`).Scan(&id)
	if !errors.Is(err, pgx.ErrNoRows) || count(tx) != 2 {
		t.Fatalf("statement without rows = %v with %d rows, want no rows and its write kept", err, count(tx))
	}
	if err := SavepointQueryRow(ctx, pool, `SELECT 4`).Scan(&id); err != nil || id != 4 {
		t.Fatalf("pool statement = (%d, %v), want 4", id, err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if rows := count(pool); rows != 2 {
		t.Fatalf("committed rows = %d, want the earlier row and the kept write", rows)
	}
}
