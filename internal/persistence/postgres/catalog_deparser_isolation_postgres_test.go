package postgres

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// A cheap deparser is a deterministic probe of SQL qualification order, not a
// race against DROP SCHEMA. It refuses every foreign OID before delegating to
// the real catalog implementation for owned constraints.
func TestPostgresStoredSizeDeparserOnlySeesOwnedConstraints(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	url := storageReviewDatabaseURL(t)
	owned, _ := isolatedPools(t, ctx, url)
	foreign, _ := isolatedPools(t, ctx, url)
	for _, pool := range []*pgxpool.Pool{owned, foreign} {
		if _, err := pool.Exec(ctx, `
CREATE TABLE catalog_probe (id bigint PRIMARY KEY, value text CONSTRAINT catalog_probe_size CHECK (octet_length(value) <= 16), count integer CHECK (count >= 0));
CREATE TABLE catalog_child (id bigint REFERENCES catalog_probe(id));
CREATE INDEX catalog_child_id_idx ON catalog_child(id) WHERE id IS NOT NULL`); err != nil {
			t.Fatal(err)
		}
	}
	tx, err := owned.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var schema string
	if err := tx.QueryRow(ctx, `SELECT current_schema()`).Scan(&schema); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx, `SET LOCAL search_path = `+pgx.Identifier{schema}.Sanitize()+`, pg_catalog`); err != nil {
		t.Fatal(err)
	}
	// Make every catalog row available to the qualification list; the very cheap
	// shadow function then exposes the order PostgreSQL may legally choose.
	if _, err := tx.Exec(ctx, `SET LOCAL enable_indexscan = off; SET LOCAL enable_bitmapscan = off`); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx, `
CREATE FUNCTION pg_get_constraintdef(oid) RETURNS text LANGUAGE plpgsql STABLE COST 0.000001 AS $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint WHERE oid = $1 AND connamespace = current_schema()::regnamespace) THEN
    RAISE EXCEPTION 'deparser received foreign constraint OID %', $1;
  END IF;
  RETURN pg_catalog.pg_get_constraintdef($1);
END
$$`); err != nil {
		t.Fatal(err)
	}
	read := func(query string) ([]string, error) {
		rows, err := tx.Query(ctx, query)
		if err != nil {
			return nil, err
		}
		return pgx.CollectRows(rows, pgx.RowTo[string])
	}
	const original = `
SELECT conname FROM pg_constraint
WHERE contype = 'c' AND connamespace = current_schema()::regnamespace
  AND pg_get_constraintdef(oid) LIKE '%octet_length(%'
ORDER BY conname`
	for _, query := range []struct{ name, sql string }{{"original", original}, {"owned", storedSizeCheckNamesSQL}} {
		plan, err := read("EXPLAIN (VERBOSE, COSTS OFF) " + query.sql)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s catalog plan:\n%s", query.name, strings.Join(plan, "\n"))
	}
	// A savepoint lets the expected negative control fail without aborting the
	// transaction in which the fixed query must subsequently succeed.
	negative, err := tx.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := negative.Query(ctx, original)
	if err == nil {
		_, err = pgx.CollectRows(rows, pgx.RowTo[string])
	}
	var rejected *pgconn.PgError
	if !errors.As(err, &rejected) || !strings.Contains(rejected.Message, "deparser received foreign constraint OID") {
		t.Fatalf("original query did not expose its foreign deparse: %v", err)
	}
	t.Logf("original qualification order rejected: %s", rejected.Message)
	if err := negative.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	names, err := read(storedSizeCheckNamesSQL)
	if err != nil || !slices.Equal(names, []string{"catalog_probe_size"}) {
		t.Fatalf("owned stored-size constraints = %v, %v", names, err)
	}

	// The other deparser in a catalog predicate is correlated with an owned FK.
	// Its index relation is already bound by that join; it must not inspect a
	// foreign index even when its function is made artificially cheap too.
	if _, err := tx.Exec(ctx, `
CREATE FUNCTION pg_get_expr(pg_node_tree, oid) RETURNS text LANGUAGE plpgsql STABLE COST 0.000001 AS $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE oid = $2 AND relnamespace = current_schema()::regnamespace) THEN
    RAISE EXCEPTION 'deparser received foreign index relation OID %', $2;
  END IF;
  RETURN pg_catalog.pg_get_expr($1, $2);
END
$$`); err != nil {
		t.Fatal(err)
	}
	plan, err := read("EXPLAIN (VERBOSE, COSTS OFF) " + foreignKeyProbeIndexesSQL)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("owned foreign-key index plan:\n%s", strings.Join(plan, "\n"))
	rows, err = tx.Query(ctx, foreignKeyProbeIndexesSQL)
	if err != nil {
		t.Fatal(err)
	}
	type probe struct {
		Name, Child, Parent string
		Covered             bool
		Indexes, Guards     []string
	}
	probes, err := pgx.CollectRows(rows, pgx.RowToStructByPos[probe])
	if err != nil || len(probes) != 1 || probes[0].Name != "catalog_child_id_fkey" || !probes[0].Covered {
		t.Fatalf("owned foreign-key index deparsing = %+v, %v", probes, err)
	}
}
