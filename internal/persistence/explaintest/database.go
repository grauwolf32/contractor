package explaintest

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Database returns a pool on a new database of its own on the server of
// databaseURL, for a fixture whose reads a test budgets, and drops the
// database when the test ends. prepare creates the schema in it, normally by
// applying the migrations; Database then turns autovacuum off for every table
// of the current schema, so the statistics and the index entries the test
// measures are those its own ANALYZE and VACUUM leave. Creating the database
// needs the CREATEDB privilege; without it the test fails.
// An empty databaseURL uses CONTRACTOR_TEST_DATABASE_URL; when neither is set,
// the test skips before connecting. Configured but unavailable databases fail.
//
// Reads count the entries of row versions that VACUUM has not removed yet
// (see the package documentation), and VACUUM keeps every version that a
// snapshot older than its deletion may still see. PostgreSQL computes that
// horizon for an ordinary table from the sessions of the table's own database
// only. In the shared test database, whose schemas belong to the tests of
// every package running in parallel, any test's open transaction keeps a
// fixture's superseded versions in its indexes, so an exact budget fails for
// a reason the test does not check (V414-038). In a database of its own,
// VACUUM removes them.
//
// So that the budgets are measured under that condition rather than assuming
// it, Database holds a snapshot open in databaseURL's database from before
// prepare until the test ends. If the fixture's database shared that
// database's horizon, every budget a superseded version can exceed would fail
// on every run instead of only when another test happens to overlap.
func Database(
	t testing.TB, ctx context.Context, databaseURL string, prepare func(context.Context, *pgxpool.Pool) error,
) *pgxpool.Pool {
	t.Helper()
	if databaseURL == "" {
		databaseURL = os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
		if databaseURL == "" {
			t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
		}
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal("invalid PostgreSQL test configuration")
	}
	admin, err := pgx.ConnectConfig(ctx, config.ConnConfig.Copy())
	if err != nil {
		t.Fatal("configured PostgreSQL test database is unavailable:", err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		_ = admin.Close(context.WithoutCancel(ctx))
		t.Fatal(err)
	}
	name := "contractor_budget_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{name}.Sanitize()
	// template0 accepts no connections, so no session can block the copy.
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+identifier+" TEMPLATE template0"); err != nil {
		_ = admin.Close(context.WithoutCancel(ctx))
		t.Fatalf("create the budget fixture's database (the test role needs CREATEDB): %v", err)
	}
	// Cleanups run last in, first out: the pool closes before the database
	// is dropped. DROP DATABASE waits for an immediate checkpoint of the
	// whole server, which takes seconds while other packages load it; the
	// deadline owns only that wait, and a database left behind fails the
	// test rather than accumulate on the server.
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		defer func() { _ = admin.Close(cleanup) }()
		if _, err := admin.Exec(cleanup, "ROLLBACK"); err != nil {
			t.Errorf("end the snapshot held beside the budget fixture: %v", err)
		}
		if _, err := admin.Exec(cleanup, "DROP DATABASE IF EXISTS "+identifier+" WITH (FORCE)"); err != nil {
			t.Errorf("drop the budget fixture's database %s: %v", name, err)
		}
	})
	if _, err := admin.Exec(ctx, "BEGIN ISOLATION LEVEL REPEATABLE READ"); err != nil {
		t.Fatal(err)
	}
	// The first statement takes the transaction's snapshot.
	if _, err := admin.Exec(ctx, "SELECT 1"); err != nil {
		t.Fatal(err)
	}
	config.ConnConfig.Database = name
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	if err := prepare(ctx, pool); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, disableAutovacuumSQL); err != nil {
		t.Fatal(err)
	}
	return pool
}
