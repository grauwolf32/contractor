package explaintest

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// supersededFixture is a table of 100 rows indexed on k, whose rows are all
// deleted and inserted again, so the index holds an entry of a deleted
// version beside each live one.
const supersededFixture = `
CREATE TABLE items (id integer PRIMARY KEY, k integer NOT NULL);
CREATE INDEX items_k_idx ON items (k);
INSERT INTO items SELECT i, i FROM generate_series(1, 100) AS i;
DELETE FROM items;
INSERT INTO items SELECT i, i FROM generate_series(1, 100) AS i`

// TestPostgresDatabaseKeepsBudgetsFromOtherSessionsSnapshots builds the same
// fixture of superseded rows in a database of its own (Database) and in a
// schema of the shared test database, vacuums both and reads 80 keys through
// the index (V414-038). In its own database the scan reads their 80 live
// entries only, although Database holds a snapshot in the shared database
// that is older than every deletion; in the shared database that snapshot
// keeps the 80 deleted entries, which the scan reads too, so the
// condition the budgets are measured under is in force. The database's tables
// have autovacuum off, and the database is gone once its test ends.
func TestPostgresDatabaseKeepsBudgetsFromOtherSessionsSnapshots(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	readKeys := func(t *testing.T, pool *pgxpool.Pool) int64 {
		t.Helper()
		if _, err := pool.Exec(ctx, `VACUUM (INDEX_CLEANUP ON, ANALYZE) items`); err != nil {
			t.Fatal(err)
		}
		var reads int64
		err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
			if _, err := tx.Exec(ctx, `SET LOCAL enable_seqscan = off; SET LOCAL enable_bitmapscan = off`); err != nil {
				return err
			}
			// Inside the histogram's end bins, so planning reads no extremes.
			analysis, err := Explain(ctx, tx, `SELECT id FROM items WHERE k BETWEEN 11 AND 90`)
			if err != nil {
				return err
			}
			if !analysis.Plan.Uses("items_k_idx") || analysis.Plan.Rows != 80 {
				t.Fatalf("the keys were not read through items_k_idx:\n%s", analysis.Plan)
			}
			reads = analysis.Reads.Of("items")
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return reads
	}
	var name string
	t.Run("fixture", func(t *testing.T) {
		own := Database(t, ctx, "", func(ctx context.Context, pool *pgxpool.Pool) error {
			_, err := pool.Exec(ctx, `CREATE TABLE settings (id integer PRIMARY KEY)`)
			return err
		})
		if _, err := own.Exec(ctx, supersededFixture); err != nil {
			t.Fatal(err)
		}
		shared := isolatedPool(t, ctx, databaseURL)
		if _, err := shared.Exec(ctx, supersededFixture); err != nil {
			t.Fatal(err)
		}
		if reads := readKeys(t, own); reads != 80 {
			t.Errorf("its own database read %d index entries, want the 80 live ones", reads)
		}
		if reads := readKeys(t, shared); reads != 160 {
			t.Errorf("the shared database read %d index entries, want the 80 live and the 80 deleted ones the held snapshot keeps", reads)
		}
		// Tables the schema had before Database returned have autovacuum
		// off; items came later.
		var autovacuumed []string
		if err := own.QueryRow(ctx, `
SELECT coalesce(array_agg(relname::text ORDER BY relname), '{}') FROM pg_class
 WHERE relkind = 'r' AND relnamespace = current_schema()::regnamespace
   AND NOT coalesce('autovacuum_enabled=off' = ANY (reloptions), false)`).Scan(&autovacuumed); err != nil {
			t.Fatal(err)
		}
		if len(autovacuumed) != 1 || autovacuumed[0] != "items" {
			t.Errorf("tables with autovacuum on = %v, want only items", autovacuumed)
		}
		if err := own.QueryRow(ctx, `SELECT current_database()`).Scan(&name); err != nil {
			t.Fatal(err)
		}
	})
	admin, err := pgx.Connect(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer admin.Close(context.WithoutCancel(ctx))
	var left bool
	if err := admin.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = $1)`, name).Scan(&left); err != nil || left || name == "" {
		t.Errorf("database %q left after its test = (%t, %v)", name, left, err)
	}
}
