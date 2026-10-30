package postgres

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// tableFunctionsSQL lists every function that a table definition of the
// current schema makes PostgreSQL evaluate while pg_restore creates tables,
// loads their rows and builds their indexes (PER-R46): the functions that
// column defaults and generated columns (pg_attrdef), CHECK constraints of
// tables and domains (pg_constraint), and index expressions and predicates
// (pg_index) call, and then the functions that those call. PostgreSQL records
// the callees of a SQL-standard body in pg_depend; a body kept as text records
// none, so the walk stops there and the check rejects that function instead.
// Built-in functions of pg_catalog are left out.
const tableFunctionsSQL = `
WITH RECURSIVE definition(catalog, oid, via) AS (
    SELECT 'pg_attrdef'::regclass, attrdef.oid, format('column %s.%s', attrdef.adrelid::regclass, attribute.attname)
      FROM pg_attrdef AS attrdef
      JOIN pg_class AS relation ON relation.oid = attrdef.adrelid
      JOIN pg_attribute AS attribute
        ON attribute.attrelid = attrdef.adrelid AND attribute.attnum = attrdef.adnum
     WHERE relation.relnamespace = current_schema()::regnamespace
    UNION ALL
    SELECT 'pg_constraint'::regclass, constraint_.oid, format('constraint %s of %s', constraint_.conname,
           coalesce(nullif(constraint_.conrelid, 0)::regclass::text, constraint_.contypid::regtype::text))
      FROM pg_constraint AS constraint_
     WHERE constraint_.connamespace = current_schema()::regnamespace
    UNION ALL
    SELECT 'pg_class'::regclass, index_.indexrelid, format('index %s', index_.indexrelid::regclass)
      FROM pg_index AS index_
      JOIN pg_class AS relation ON relation.oid = index_.indexrelid
     WHERE relation.relnamespace = current_schema()::regnamespace
), reached(function, via) AS (
    SELECT dependency.refobjid, definition.via
      FROM definition
      JOIN pg_depend AS dependency
        ON dependency.classid = definition.catalog AND dependency.objid = definition.oid
       AND dependency.refclassid = 'pg_proc'::regclass
    UNION
    SELECT dependency.refobjid, reached.via
      FROM reached
      JOIN pg_depend AS dependency
        ON dependency.classid = 'pg_proc'::regclass AND dependency.objid = reached.function
       AND dependency.refclassid = 'pg_proc'::regclass
)
SELECT function.oid::regprocedure::text, language.lanname::text, function.prosqlbody IS NOT NULL,
       coalesce(function.proconfig, '{}'), string_agg(DISTINCT reached.via, '; ' ORDER BY reached.via)
  FROM reached
  JOIN pg_proc AS function ON function.oid = reached.function
  JOIN pg_language AS language ON language.oid = function.prolang
 WHERE function.pronamespace <> 'pg_catalog'::regnamespace
 GROUP BY function.oid, language.lanname
 ORDER BY function.oid::regprocedure::text COLLATE "C"`

// tableFunction is one function that a table definition evaluates, directly
// or through other functions.
type tableFunction struct {
	signature, language string
	sqlStandardBody     bool
	config              []string
	via                 string
}

// restorable reports whether pg_restore can run the function with its empty
// search_path (PER-R46): a SQL-standard body bound its names when it was
// created, and a pinned pg_catalog search path lets the body name only
// built-in objects.
func (f tableFunction) restorable() bool {
	return f.language == "sql" && f.sqlStandardBody || slices.Contains(f.config, "search_path=pg_catalog")
}

// unrestorableTableFunctions returns the functions that the table definitions
// of pool's schema evaluate and one message per function among them that
// pg_restore cannot run.
func unrestorableTableFunctions(ctx context.Context, pool *pgxpool.Pool) ([]tableFunction, []string, error) {
	rows, err := pool.Query(ctx, tableFunctionsSQL)
	if err != nil {
		return nil, nil, err
	}
	functions, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (tableFunction, error) {
		var function tableFunction
		err := row.Scan(&function.signature, &function.language, &function.sqlStandardBody, &function.config, &function.via)
		return function, err
	})
	if err != nil {
		return nil, nil, err
	}
	var problems []string
	for _, function := range functions {
		if !function.restorable() {
			problems = append(problems, fmt.Sprintf(
				"%s (LANGUAGE %s) is evaluated by %s, but its body is text that resolves names through the session search_path, which pg_restore empties (PER-R46); give a LANGUAGE sql function a BEGIN ATOMIC or RETURN body, or pin search_path = pg_catalog",
				function.signature, function.language, function.via))
		}
	}
	return functions, problems, nil
}

// TestPostgresTableDefinitionFunctionsAreRestorable checks every function that
// a table definition of the migrated schema evaluates, so that a later
// migration cannot add a backup that pg_restore fails to load (PER-R46).
func TestPostgresTableDefinitionFunctionsAreRestorable(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, databaseURL)
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	functions, problems, err := unrestorableTableFunctions(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	if len(functions) < 10 {
		t.Fatalf("found only %d functions that table definitions evaluate; the catalog query is broken", len(functions))
	}
	for _, function := range functions {
		t.Logf("%s, LANGUAGE %s, SQL-standard body %v, config %v: %s",
			function.signature, function.language, function.sqlStandardBody, function.config, function.via)
	}
	for _, problem := range problems {
		t.Error(problem)
	}
}

// TestPostgresTableDefinitionFunctionCheckFindsTextBodies shows that the check
// above can fail: on a schema of its own it reaches every kind of table
// definition, follows a SQL-standard body to the text body it calls, and
// rejects exactly the text bodies that do not pin their search path.
func TestPostgresTableDefinitionFunctionCheckFindsTextBodies(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, databaseURL)
	if _, err := pool.Exec(ctx, `
CREATE FUNCTION probe_text_leaf(value integer) RETURNS boolean
    LANGUAGE sql IMMUTABLE AS 'SELECT value > 0';
CREATE FUNCTION probe_text_caller(value integer) RETURNS boolean
    LANGUAGE sql IMMUTABLE AS 'SELECT probe_text_leaf(value)';
CREATE FUNCTION probe_atomic_caller(value integer) RETURNS boolean
    LANGUAGE sql IMMUTABLE RETURN probe_text_leaf(value);
CREATE FUNCTION probe_atomic(value integer) RETURNS integer
    LANGUAGE sql IMMUTABLE BEGIN ATOMIC SELECT value + 1; END;
CREATE FUNCTION probe_pinned(value integer) RETURNS boolean
    LANGUAGE plpgsql IMMUTABLE SET search_path = pg_catalog AS 'BEGIN RETURN value > 0; END';
CREATE FUNCTION probe_unpinned(value integer) RETURNS boolean
    LANGUAGE plpgsql IMMUTABLE AS 'BEGIN RETURN value > 0; END';
CREATE FUNCTION probe_default() RETURNS integer
    LANGUAGE sql IMMUTABLE AS 'SELECT 1';
CREATE DOMAIN probe_positive AS integer CHECK (probe_unpinned(VALUE));
CREATE TABLE probe (
    checked integer CHECK (probe_text_caller(checked)),
    nested integer CHECK (probe_atomic_caller(nested)),
    pinned integer CHECK (probe_pinned(pinned)),
    generated integer GENERATED ALWAYS AS (probe_atomic(checked)) STORED,
    defaulted integer DEFAULT probe_default(),
    typed probe_positive
);
CREATE INDEX probe_expression ON probe (probe_atomic(nested)) WHERE probe_pinned(pinned);`); err != nil {
		t.Fatal(err)
	}
	functions, problems, err := unrestorableTableFunctions(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	var reached []string
	for _, function := range functions {
		reached = append(reached, function.signature+" <- "+function.via)
	}
	want := []string{
		"probe_atomic(integer) <- column probe.generated; index probe_expression",
		"probe_atomic_caller(integer) <- constraint probe_nested_check of probe",
		"probe_default() <- column probe.defaulted",
		"probe_pinned(integer) <- constraint probe_pinned_check of probe; index probe_expression",
		"probe_text_caller(integer) <- constraint probe_checked_check of probe",
		"probe_text_leaf(integer) <- constraint probe_nested_check of probe",
		"probe_unpinned(integer) <- constraint probe_positive_check of probe_positive",
	}
	if !slices.Equal(reached, want) {
		t.Fatalf("reached functions:\n%s\nwant:\n%s", strings.Join(reached, "\n"), strings.Join(want, "\n"))
	}
	var rejected []string
	for _, problem := range problems {
		rejected = append(rejected, problem[:strings.IndexByte(problem, ' ')])
	}
	if want := []string{"probe_default()", "probe_text_caller(integer)", "probe_text_leaf(integer)", "probe_unpinned(integer)"}; !slices.Equal(rejected, want) {
		t.Fatalf("rejected %v, want %v", rejected, want)
	}
}
