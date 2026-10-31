package explaintest

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// preparedFixture is a table of 1,000 rows whose k takes each of 100 values
// ten times, with an index on k.
const preparedFixture = `
CREATE TABLE items (id integer PRIMARY KEY, k integer NOT NULL, pad text NOT NULL);
CREATE INDEX items_k_idx ON items (k);
INSERT INTO items SELECT i, i % 100, md5(i::text) FROM generate_series(1, 1000) AS i;
CREATE TABLE other (id integer PRIMARY KEY);
ANALYZE items, other`

// TestPostgresExplainPreparedCountsTheRowsOfEachPlan runs the same statement
// in both plan cache modes on one connection. An optional filter, folded
// away when the custom plan knows the argument, lets the custom plan probe
// the index for the ten matching entries while the generic plan scans all
// 1,000 rows; Reads counts exactly those, and Explain counts as the custom
// plan does. The analysis leaves neither a row change, nor the plan cache
// mode, nor the prepared statement behind, and a failing statement leaves the
// enclosing transaction usable.
func TestPostgresExplainPreparedCountsTheRowsOfEachPlan(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	pool := isolatedPool(t, ctx, databaseURL)
	if _, err := pool.Exec(ctx, preparedFixture); err != nil {
		t.Fatal(err)
	}
	const optional = `SELECT id FROM items WHERE ($1::integer IS NULL OR k = $1)`
	for mode, want := range map[Mode]int64{Custom: 10, Generic: 1000} {
		analysis, err := ExplainPrepared(ctx, pool, mode, optional, 7)
		if err != nil {
			t.Fatal(err)
		}
		if got := analysis.Reads.Of("items"); got != want || analysis.Reads.Of("other") != 0 {
			t.Errorf("%s plan read %d items and %d other rows, want %d and none:\n%s",
				mode, got, analysis.Reads.Of("other"), want, analysis.Plan)
		}
	}
	analysis, err := Explain(ctx, pool, optional, 7)
	if err != nil {
		t.Fatal(err)
	}
	if got := analysis.Reads.Of("items"); got != 10 {
		t.Errorf("Explain read %d items, want the custom plan's 10:\n%s", got, analysis.Plan)
	}
	// A filter reads every row it discards.
	analysis, err = ExplainPrepared(ctx, pool, Custom, `SELECT count(*) FROM items WHERE pad = $1`, "none")
	if err != nil {
		t.Fatal(err)
	}
	if got := analysis.Reads.Of("items"); got != 1000 {
		t.Errorf("a filtered scan read %d items, want 1000:\n%s", got, analysis.Plan)
	}

	// Rows a parallel worker reads would not reach the counters.
	err = pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `
SET LOCAL max_parallel_workers_per_gather = 4; SET LOCAL parallel_setup_cost = 0;
SET LOCAL parallel_tuple_cost = 0; SET LOCAL min_parallel_table_scan_size = 0`); err != nil {
			return err
		}
		analysis, err := ExplainPrepared(ctx, tx, Custom, `SELECT count(*) FROM items WHERE pad = $1`, "none")
		if err != nil {
			return err
		}
		gathers := 0
		for _, node := range analysis.Plan.Nodes() {
			if strings.HasPrefix(node.Type, "Gather") {
				gathers++
			}
		}
		if got := analysis.Reads.Of("items"); got != 1000 || gathers != 0 {
			return fmt.Errorf("a parallel-eligible scan read %d items under %d Gather nodes, want 1000 and none:\n%+v",
				got, gathers, analysis.Plan)
		}
		var workers string
		if err := tx.QueryRow(ctx, `SELECT current_setting('max_parallel_workers_per_gather')`).Scan(&workers); err != nil {
			return err
		}
		if workers != "4" {
			return fmt.Errorf("max_parallel_workers_per_gather is %s after the count, want the transaction's 4", workers)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	err = pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		if _, err := ExplainPrepared(ctx, tx, Generic, `UPDATE items SET pad = $1 WHERE k = $2`, "changed", 7); err != nil {
			return err
		}
		if _, err := ExplainPrepared(ctx, tx, Custom, `UPDATE items SET k = $1 WHERE id = $2`, "not a number", 1); err == nil {
			return fmt.Errorf("an argument of the wrong type was accepted")
		}
		var changed int
		var mode string
		var statements int
		if err := tx.QueryRow(ctx, `
SELECT (SELECT count(*) FROM items WHERE pad = 'changed'), current_setting('plan_cache_mode'),
       (SELECT count(*) FROM pg_prepared_statements WHERE name LIKE 'explaintest\_%')`,
		).Scan(&changed, &mode, &statements); err != nil {
			return err
		}
		// The failed analysis left its statement on the session.
		if changed != 0 || mode != "auto" || statements != 1 {
			return fmt.Errorf("after the analyses %d rows changed, plan_cache_mode is %s and %d statements remain",
				changed, mode, statements)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// isolatedPool returns a one-connection pool on a new schema that the test
// drops when it ends.
func isolatedPool(t *testing.T, ctx context.Context, databaseURL string) *pgxpool.Pool {
	t.Helper()
	pool := postgrestest.NewPool(t, ctx, databaseURL, postgrestest.Options{
		Configure: func(poolConfig *pgxpool.Config) {
			poolConfig.MaxConns = 1
		},
	})
	return pool
}
