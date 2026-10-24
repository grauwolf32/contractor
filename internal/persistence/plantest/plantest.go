// Package plantest measures the work one statement does on a migrated
// schema, for row-budget tests of production SQL. Explain prepares the
// statement under a forced plan cache mode, so a test sees both the custom
// plan PostgreSQL makes for a statement's first executions and the generic
// plan a pooled connection's prepared statement may switch to later, runs it
// with EXPLAIN ANALYZE inside a transaction that it rolls back, and reports
// how many rows the execution read from each table: the rows its sequential
// scans read and the entries its index scans read, taken from the
// transaction's own statistics counters, which are exact where EXPLAIN
// rounds each node's rows to a per-loop average. For a fixed fixture and
// fixed statistics that count is deterministic; it never measures time.
// An index's count is the entries the index returned, so an entry that a
// later column's index condition rejects inside the index is not in it, and
// an entry of a superseded row version that VACUUM has not removed yet is.
// Planning may probe an index for a column's current extremes, so a budget
// allows a few rows beyond the execution's own. Production code never
// imports this package.
package plantest

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Mode is a plan_cache_mode under which Explain plans the statement.
type Mode string

const (
	// Custom plans the statement with its parameter values, as PostgreSQL
	// does for the first executions of a prepared statement.
	Custom Mode = "force_custom_plan"
	// Generic plans the statement without its parameter values, as
	// PostgreSQL may for later executions of a prepared statement.
	Generic Mode = "force_generic_plan"
)

// Modes are both plan cache modes a pooled prepared statement can run in.
var Modes = []Mode{Custom, Generic}

// Scan is one executed plan node that reads a relation or an index; its row
// counts are EXPLAIN's per-loop averages, rounded.
type Scan struct {
	NodeType  string `json:"Node Type"`
	Relation  string `json:"Relation Name"`
	IndexName string `json:"Index Name"`
	IndexCond string `json:"Index Cond"`
	Filter    string `json:"Filter"`
	// Rows is the rows the node returned in each loop; RemovedByFilter and
	// RemovedByRecheck are the rows it read and then discarded in each loop.
	Rows             float64 `json:"Actual Rows"`
	Loops            float64 `json:"Actual Loops"`
	RemovedByFilter  float64 `json:"Rows Removed by Filter"`
	RemovedByRecheck float64 `json:"Rows Removed by Index Recheck"`
}

// Plan is one executed plan.
type Plan struct {
	// JSON is the EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) output, for a
	// failure message or a test log.
	JSON  string
	scans []Scan
	read  map[string]int64
}

// IndexScans returns the plan nodes that scan an index whose name starts
// with prefix, a bitmap index scan included, which names no relation.
func (p Plan) IndexScans(prefix string) []Scan {
	var result []Scan
	for _, scan := range p.scans {
		if scan.IndexName != "" && strings.HasPrefix(scan.IndexName, prefix) {
			result = append(result, scan)
		}
	}
	return result
}

// Read returns the rows that executing the plan read from table: the rows
// its sequential scans read plus the entries its scans of the table's
// indexes read, whether a filter kept them or not.
func (p Plan) Read(table string) int64 {
	return p.read[table]
}

// readCountersSQL returns, for every table of the current schema, the rows
// the current transaction's sequential scans read and the entries its index
// scans read so far.
const readCountersSQL = `
SELECT table_class.relname::text,
       pg_stat_get_xact_tuples_returned(table_class.oid)
         + COALESCE(sum(pg_stat_get_xact_tuples_returned(index.indexrelid)), 0)::bigint
  FROM pg_class AS table_class
  LEFT JOIN pg_index AS index ON index.indrelid = table_class.oid
 WHERE table_class.relkind IN ('r', 'p')
   AND table_class.relnamespace = current_schema()::regnamespace
 GROUP BY table_class.oid, table_class.relname`

func readCounters(ctx context.Context, tx pgx.Tx) (map[string]int64, error) {
	rows, err := tx.Query(ctx, readCountersSQL)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := map[string]int64{}
	for rows.Next() {
		var table string
		var count int64
		if err := rows.Scan(&table, &count); err != nil {
			return nil, err
		}
		result[table] = count
	}
	return result, rows.Err()
}

// statementName is the prepared statement Explain creates and deallocates.
const statementName = "plantest_statement"

// Explain executes sql with args as a prepared statement planned in mode and
// returns its executed plan. It runs on one connection of pool, inside a
// transaction it rolls back, so a statement that writes leaves no change;
// the arguments are passed as SQL literals because EXECUTE takes no bound
// parameters. It fails t on any error.
func Explain(t testing.TB, ctx context.Context, pool *pgxpool.Pool, mode Mode, sql string, args ...any) Plan {
	t.Helper()
	literals := make([]string, len(args))
	for index, arg := range args {
		literal, err := Literal(arg)
		if err != nil {
			t.Fatalf("argument $%d: %v", index+1, err)
		}
		literals[index] = literal
	}
	conn, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Release()
	tx, err := conn.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if _, err := tx.Exec(ctx, "SET LOCAL plan_cache_mode = "+string(mode)); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx, "PREPARE "+statementName+" AS "+sql); err != nil {
		t.Fatalf("prepare: %v\n%s", err, sql)
	}
	// A prepared statement belongs to the session, not to the transaction.
	defer func() { _, _ = conn.Exec(context.WithoutCancel(ctx), "DEALLOCATE "+statementName) }()
	execute := "EXECUTE " + statementName
	if len(literals) != 0 {
		execute += "(" + strings.Join(literals, ", ") + ")"
	}
	before, err := readCounters(ctx, tx)
	if err != nil {
		t.Fatal(err)
	}
	var raw string
	if err := tx.QueryRow(ctx,
		"EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, SUMMARY OFF, FORMAT JSON) "+execute,
	).Scan(&raw); err != nil {
		t.Fatalf("explain: %v\n%s", err, sql)
	}
	after, err := readCounters(ctx, tx)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	var roots []struct {
		Plan json.RawMessage `json:"Plan"`
	}
	if err := json.Unmarshal([]byte(raw), &roots); err != nil || len(roots) != 1 {
		t.Fatalf("decode plan: %v\n%s", err, raw)
	}
	plan := Plan{JSON: raw, read: map[string]int64{}}
	for table, count := range after {
		if delta := count - before[table]; delta != 0 {
			plan.read[table] = delta
		}
	}
	if err := collect(roots[0].Plan, &plan.scans); err != nil {
		t.Fatalf("decode plan: %v\n%s", err, raw)
	}
	return plan
}

// collect appends node and its descendants that read a relation or an
// index to scans.
func collect(raw json.RawMessage, scans *[]Scan) error {
	var node struct {
		Scan
		Plans []json.RawMessage `json:"Plans"`
	}
	if err := json.Unmarshal(raw, &node); err != nil {
		return err
	}
	if node.Relation != "" || node.IndexName != "" {
		*scans = append(*scans, node.Scan)
	}
	for _, child := range node.Plans {
		if err := collect(child, scans); err != nil {
			return err
		}
	}
	return nil
}

// Literal renders arg as a SQL literal: nil and nil pointers as NULL, strings
// quoted, integers and booleans as such, string slices as text[] and times as
// timestamptz. A value of a named type renders as its underlying kind.
func Literal(arg any) (string, error) {
	if arg == nil {
		return "NULL", nil
	}
	if value, ok := arg.(time.Time); ok {
		return quote(value.UTC().Format(time.RFC3339Nano)) + "::timestamptz", nil
	}
	value := reflect.ValueOf(arg)
	switch value.Kind() {
	case reflect.Pointer:
		if value.IsNil() {
			return "NULL", nil
		}
		return Literal(value.Elem().Interface())
	case reflect.String:
		return quote(value.String()), nil
	case reflect.Bool:
		return strconv.FormatBool(value.Bool()), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(value.Int(), 10), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return strconv.FormatUint(value.Uint(), 10), nil
	case reflect.Slice:
		if value.Type().Elem().Kind() != reflect.String {
			break
		}
		if value.IsNil() {
			return "NULL", nil
		}
		elements := make([]string, value.Len())
		for index := range elements {
			elements[index] = quote(value.Index(index).String())
		}
		if len(elements) == 0 {
			return "'{}'::text[]", nil
		}
		return "ARRAY[" + strings.Join(elements, ", ") + "]::text[]", nil
	}
	return "", fmt.Errorf("unsupported argument type %T", arg)
}

// quote renders text as a standard-conforming SQL string literal.
func quote(text string) string {
	return "'" + strings.ReplaceAll(text, "'", "''") + "'"
}
