// Package explaintest measures, against a real PostgreSQL, the work a SQL
// statement does: the plan PostgreSQL ran, with the shared blocks it touched,
// the rows it read from each table and the row triggers it called. Explain
// runs the statement under EXPLAIN ANALYZE in a savepoint that it rolls back,
// so analyzing a write leaves nothing behind; ExplainPrepared does the same
// for the statement prepared under a forced plan cache mode, so a test sees
// both the custom plan PostgreSQL makes for a pooled prepared statement's
// first executions and the generic plan it may switch to later; Probe
// analyzes every statement a store sends through it before it runs the
// statement for real. Database gives a measured fixture a database of its
// own. RequireColumnScopedGuards holds the column-scoped row guards of a table
// to the columns they list in UPDATE OF (PER-R22); with the same lexer,
// SplitStatements reads SQL text, such as a migration file, and ReadBody the
// tables a routine body writes and the functions it calls.
//
// The rows a statement read from a table (Reads) are the rows the table's
// sequential scans returned plus the entries the table's indexes returned,
// as the transaction's own statistics counters count them. It is the one
// count of rows read that the package offers, and row budgets use it. The
// count is exact where EXPLAIN rounds each node's rows to a per-loop average,
// so a row that an inner scan reads in fewer than half of its loops counts
// although the plan reports none, and for a fixed fixture it is
// deterministic; it never measures time. Rows a
// parallel worker reads never reach the counters, so a counted statement is
// planned without parallel workers. The
// count includes rows a filter discarded later, the entries of an index-only
// scan and every entry a bitmap index scan returned, and also an entry of a
// row version that is dead but that VACUUM has not removed yet, or, for a
// B-tree, that no earlier scan has marked dead. A fixture that updates or
// deletes rows is therefore built in a database of its own (Database), where
// no other test's snapshot keeps those versions from its VACUUM; a fixture
// that only inserts may share the test database. The count excludes an entry
// that a later column's index condition rejected inside the index and the
// other rows of a lossy bitmap page. Planning may probe an index for a
// column's current extremes, a custom plan's for its arguments and any plan's
// for a join estimate, and those entries count too, so a budget allows a few
// rows beyond the execution's own.
//
// EXPLAIN reports the triggers of the statement's own tables, deferred ones
// included once SET CONSTRAINTS ALL IMMEDIATE has made them run at the end of
// the statement, but not the triggers of statements a trigger runs, and only
// triggers that were called: a trigger skipped by its UPDATE OF columns or
// its WHEN condition is absent. Production code never imports this package.
package explaintest

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"unicode"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// Beginner starts the savepoint (in a transaction) or the transaction (on a
// pool) that Explain rolls back.
type Beginner interface {
	Begin(ctx context.Context) (pgx.Tx, error)
}

// Trigger names one trigger of one relation.
type Trigger struct {
	Relation, Name string
}

// Analysis is what EXPLAIN ANALYZE reported for one statement.
type Analysis struct {
	Plan Node
	// Triggers counts the calls of each trigger the statement called.
	Triggers map[Trigger]int64
	// Reads counts the rows the statement read from each table.
	Reads Reads
}

// Node is one plan node. Rows and the removed counts are averages per loop,
// as EXPLAIN reports them.
type Node struct {
	Type             string  `json:"Node Type"`
	Relationship     string  `json:"Parent Relationship"`
	Relation         string  `json:"Relation Name"`
	Index            string  `json:"Index Name"`
	IndexCond        string  `json:"Index Cond"`
	Rows             float64 `json:"Actual Rows"`
	Loops            float64 `json:"Actual Loops"`
	RemovedByFilter  float64 `json:"Rows Removed by Filter"`
	RemovedByRecheck float64 `json:"Rows Removed by Index Recheck"`
	SharedHit        int64   `json:"Shared Hit Blocks"`
	SharedRead       int64   `json:"Shared Read Blocks"`
	Children         []Node  `json:"Plans"`
}

// Explain runs sql with args under EXPLAIN (ANALYZE, BUFFERS) in a savepoint
// or transaction of db that it rolls back, and returns what PostgreSQL
// reported. The statement really runs, triggers and constraint checks
// included, so it fails as the statement itself would.
func Explain(ctx context.Context, db Beginner, sql string, args ...any) (Analysis, error) {
	return analyze(ctx, db, nil, sql, args, nil)
}

// Mode is a plan_cache_mode under which ExplainPrepared plans a statement.
type Mode string

const (
	// Custom plans the statement with its arguments, as PostgreSQL does for
	// the first executions of a prepared statement.
	Custom Mode = "force_custom_plan"
	// Generic plans the statement without its arguments, as PostgreSQL may
	// for later executions of a prepared statement.
	Generic Mode = "force_generic_plan"
)

// Modes are both plan cache modes a pooled prepared statement can run in.
var Modes = []Mode{Custom, Generic}

// prepared numbers the statements ExplainPrepared prepares.
var prepared atomic.Uint64

// ExplainPrepared is Explain for sql prepared as a statement of its own and
// planned in mode. EXECUTE takes no bound parameters, so the arguments reach
// it as SQL literals that pgx renders, and the statement's parameter types
// are those PostgreSQL infers from sql. A prepared statement belongs to the
// session rather than to the transaction: ExplainPrepared deallocates it
// after a successful analysis, and a statement whose analysis failed stays
// on the session under a name no other call uses.
func ExplainPrepared(ctx context.Context, db Beginner, mode Mode, sql string, args ...any) (Analysis, error) {
	name := fmt.Sprintf("explaintest_%d", prepared.Add(1))
	execute := "EXECUTE " + name
	if len(args) > 0 {
		placeholders := make([]string, len(args))
		for index := range args {
			placeholders[index] = fmt.Sprintf("$%d", index+1)
		}
		execute += "(" + strings.Join(placeholders, ", ") + ")"
		args = append([]any{pgx.QueryExecModeSimpleProtocol}, args...)
	}
	return analyze(ctx, db,
		[]string{"SET LOCAL plan_cache_mode = " + string(mode), "PREPARE " + name + " AS " + sql},
		execute, args, []string{"DEALLOCATE " + name})
}

// analyze runs setup, then statement with args under EXPLAIN ANALYZE while
// it counts the rows read, then teardown, in a savepoint or transaction of
// db that it rolls back.
func analyze(
	ctx context.Context, db Beginner, setup []string, statement string, args []any, teardown []string,
) (Analysis, error) {
	savepoint, err := db.Begin(ctx)
	if err != nil {
		return Analysis{}, err
	}
	var output []byte
	var reads Reads
	err = func() error {
		for _, sql := range setup {
			if _, err := savepoint.Exec(ctx, sql); err != nil {
				return err
			}
		}
		var err error
		reads, err = CountReads(ctx, savepoint, func() error {
			return savepoint.QueryRow(ctx,
				`EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, SUMMARY OFF, FORMAT JSON) `+statement, args...).Scan(&output)
		})
		for _, sql := range teardown {
			if err != nil {
				break
			}
			_, err = savepoint.Exec(ctx, sql)
		}
		return err
	}()
	if rollbackErr := savepoint.Rollback(ctx); err == nil {
		err = rollbackErr
	}
	if err != nil {
		return Analysis{}, err
	}
	analysis, err := parse(output)
	analysis.Reads = reads
	return analysis, err
}

// Reads counts the rows statements read from each table of the current
// schema, as the package documentation defines them.
type Reads map[string]int64

// Of returns the rows read from table. It panics when the current schema has
// no such table, so a misspelt name cannot pass a budget.
func (r Reads) Of(table string) int64 {
	count, ok := r[table]
	if !ok {
		panic(fmt.Sprintf("explaintest: the current schema has no table %q", table))
	}
	return count
}

// readsSQL returns, for every table of the current schema, the rows the
// current transaction's sequential scans of it returned and the entries its
// indexes returned so far.
const readsSQL = `
SELECT table_class.relname::text,
       pg_stat_get_xact_tuples_returned(table_class.oid)
         + COALESCE(sum(pg_stat_get_xact_tuples_returned(index.indexrelid)), 0)::bigint
  FROM pg_class AS table_class
  LEFT JOIN pg_index AS index ON index.indrelid = table_class.oid
 WHERE table_class.relkind IN ('r', 'p')
   AND table_class.relnamespace = current_schema()::regnamespace
 GROUP BY table_class.oid, table_class.relname`

// CountReads calls run and returns the rows the statements it sent through
// tx read from each table. PostgreSQL keeps the counters per transaction, so
// run's statements must go through tx itself. The rows a parallel worker
// reads never reach the transaction's counters, so run plans without
// parallel workers and the transaction's own setting returns afterwards.
func CountReads(ctx context.Context, tx pgx.Tx, run func() error) (Reads, error) {
	var parallel string
	if err := tx.QueryRow(ctx,
		`SELECT current_setting('max_parallel_workers_per_gather'), set_config('max_parallel_workers_per_gather', '0', true)`,
	).Scan(&parallel, new(string)); err != nil {
		return nil, err
	}
	before, err := readCounters(ctx, tx)
	if err != nil {
		return nil, err
	}
	if err := run(); err != nil {
		return nil, err
	}
	after, err := readCounters(ctx, tx)
	if err != nil {
		return nil, err
	}
	if _, err := tx.Exec(ctx, `SELECT set_config('max_parallel_workers_per_gather', $1, true)`, parallel); err != nil {
		return nil, err
	}
	for table, count := range after {
		after[table] = count - before[table]
	}
	return after, nil
}

func readCounters(ctx context.Context, tx pgx.Tx) (Reads, error) {
	rows, err := tx.Query(ctx, readsSQL)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	reads := Reads{}
	for rows.Next() {
		var table string
		var count int64
		if err := rows.Scan(&table, &count); err != nil {
			return nil, err
		}
		reads[table] = count
	}
	return reads, rows.Err()
}

// ExplainGeneric returns the generic plan of sql, the one a cached prepared
// statement may reuse whatever its arguments, without running it. Only the
// simple query protocol passes the $n placeholders through unbound, so it
// takes a connection rather than a pool; the plan has no actual rows.
func ExplainGeneric(ctx context.Context, conn *pgconn.PgConn, sql string) (Node, error) {
	results, err := conn.Exec(ctx, `EXPLAIN (GENERIC_PLAN, FORMAT JSON) `+sql).ReadAll()
	if err != nil {
		return Node{}, err
	}
	if len(results) != 1 || len(results[0].Rows) != 1 || len(results[0].Rows[0]) != 1 {
		return Node{}, fmt.Errorf("EXPLAIN (GENERIC_PLAN) returned %d results, want one row", len(results))
	}
	analysis, err := parse(results[0].Rows[0][0])
	return analysis.Plan, err
}

func parse(output []byte) (Analysis, error) {
	var plans []struct {
		Plan     Node `json:"Plan"`
		Triggers []struct {
			Name     string `json:"Trigger Name"`
			Relation string `json:"Relation"`
			Calls    int64  `json:"Calls"`
		} `json:"Triggers"`
	}
	if err := json.Unmarshal(output, &plans); err != nil {
		return Analysis{}, fmt.Errorf("decode EXPLAIN output: %w", err)
	}
	if len(plans) != 1 || plans[0].Plan.Type == "" {
		return Analysis{}, fmt.Errorf("EXPLAIN output holds %d plans, want one: %s", len(plans), output)
	}
	analysis := Analysis{Plan: plans[0].Plan, Triggers: map[Trigger]int64{}}
	for _, trigger := range plans[0].Triggers {
		analysis.Triggers[Trigger{Relation: trigger.Relation, Name: trigger.Name}] += trigger.Calls
	}
	return analysis, nil
}

// Blocks returns the shared blocks the plan hit or read, its InitPlans and
// SubPlans included; planning is not counted.
func (n Node) Blocks() int64 {
	return n.SharedHit + n.SharedRead
}

// Nodes returns the plan's nodes, the root first and each node before its
// children, InitPlans and SubPlans included.
func (n Node) Nodes() []Node {
	var nodes []Node
	n.walk(func(node Node, _ int) { nodes = append(nodes, node) })
	return nodes
}

// Uses reports whether any node of the plan scans index.
func (n Node) Uses(index string) bool {
	used := false
	n.walk(func(node Node, _ int) { used = used || node.Index == index })
	return used
}

// SeqScans returns the relations the plan scans sequentially.
func (n Node) SeqScans() []string {
	var relations []string
	n.walk(func(node Node, _ int) {
		if node.Type == "Seq Scan" {
			relations = append(relations, node.Relation)
		}
	})
	return relations
}

// String renders the plan as one indented line per node with its relation,
// index and index condition, rows over all loops, rows removed and shared
// blocks, for test logs.
func (n Node) String() string {
	var lines []string
	n.walk(func(node Node, depth int) {
		line := strings.Repeat("  ", depth)
		if node.Relationship == "InitPlan" || node.Relationship == "SubPlan" {
			line += node.Relationship + ": "
		}
		line += node.Type
		if node.Relation != "" {
			line += " on " + node.Relation
		}
		if node.Index != "" {
			line += " using " + node.Index
		}
		if node.IndexCond != "" {
			line += " where " + node.IndexCond
		}
		line += fmt.Sprintf(" rows=%.0f removed=%.0f blocks=%d",
			node.Rows*node.Loops, (node.RemovedByFilter+node.RemovedByRecheck)*node.Loops, node.Blocks())
		lines = append(lines, line)
	})
	return strings.Join(lines, "\n")
}

func (n Node) walk(visit func(Node, int)) {
	var walk func(Node, int)
	walk = func(node Node, depth int) {
		visit(node, depth)
		for _, child := range node.Children {
			walk(child, depth+1)
		}
	}
	walk(n, 0)
}

// Probe is a transaction that analyzes each statement sent through Exec,
// Query or QueryRow before it runs it, and adds up the triggers each one
// called, so a store bound to it reports which triggers its operations call.
// Every statement therefore runs twice, the first time in a savepoint that is
// rolled back; a savepoint begun through the Probe probes its statements too.
// Utility statements, batches and copies are not probed.
type Probe struct {
	pgx.Tx
	t     testing.TB
	fired map[Trigger]int64
}

// NewProbe probes the statements sent through tx.
func NewProbe(t testing.TB, tx pgx.Tx) *Probe {
	return &Probe{Tx: tx, t: t, fired: map[Trigger]int64{}}
}

// Fired returns the calls of each trigger of relation since the last Reset,
// leaving out the internal triggers of foreign keys (RI_ConstraintTrigger_*).
// Those are key checks, not triggers the schema defines, and PostgreSQL calls
// them even for an unchanged key of a row inserted earlier in the same
// transaction.
func (p *Probe) Fired(relation string) map[string]int64 {
	fired := map[string]int64{}
	for trigger, calls := range p.fired {
		if trigger.Relation == relation && !strings.HasPrefix(trigger.Name, foreignKeyTriggerPrefix) {
			fired[trigger.Name] += calls
		}
	}
	return fired
}

// foreignKeyTriggerPrefix starts the name of every internal trigger that
// PostgreSQL creates for a foreign key.
const foreignKeyTriggerPrefix = "RI_ConstraintTrigger_"

// Reset forgets the triggers counted so far.
func (p *Probe) Reset() {
	clear(p.fired)
}

// Begin starts a savepoint whose statements add to the same counts.
func (p *Probe) Begin(ctx context.Context) (pgx.Tx, error) {
	savepoint, err := p.Tx.Begin(ctx)
	if err != nil {
		return nil, err
	}
	return &Probe{Tx: savepoint, t: p.t, fired: p.fired}, nil
}

// analyzable reports whether the probe explains sql: every statement except
// the named utility statements SHOW, SET and RESET, which EXPLAIN rejects and
// which call no trigger. Any other statement EXPLAIN cannot run fails the
// probe instead of being skipped, so no data change goes uncounted.
func analyzable(sql string) bool {
	for {
		sql = strings.TrimSpace(sql)
		if !strings.HasPrefix(sql, "--") {
			break
		}
		_, sql, _ = strings.Cut(sql, "\n")
	}
	end := strings.IndexFunc(sql, func(r rune) bool { return unicode.IsSpace(r) || r == ';' })
	if end < 0 {
		end = len(sql)
	}
	switch strings.ToUpper(sql[:end]) {
	case "SHOW", "SET", "RESET", "":
		return false
	}
	return true
}

func (p *Probe) analyze(ctx context.Context, sql string, args ...any) {
	p.t.Helper()
	if !analyzable(sql) {
		return
	}
	analysis, err := Explain(ctx, p.Tx, sql, args...)
	if err != nil {
		p.t.Fatalf("explain %q: %v", sql, err)
	}
	for trigger, calls := range analysis.Triggers {
		p.fired[trigger] += calls
	}
}

func (p *Probe) Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error) {
	p.analyze(ctx, sql, args...)
	return p.Tx.Exec(ctx, sql, args...)
}

func (p *Probe) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	p.analyze(ctx, sql, args...)
	return p.Tx.Query(ctx, sql, args...)
}

func (p *Probe) QueryRow(ctx context.Context, sql string, args ...any) pgx.Row {
	p.analyze(ctx, sql, args...)
	return p.Tx.QueryRow(ctx, sql, args...)
}
