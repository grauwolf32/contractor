// Package explaintest measures, against a real PostgreSQL, the work a SQL
// statement does: the plan PostgreSQL ran, with the rows each scan read and
// the shared blocks the statement touched, and the row triggers it called.
// Explain runs the statement under EXPLAIN ANALYZE in a savepoint that it
// rolls back, so analyzing a write leaves nothing behind; Probe does that
// for every statement a store sends through it before it runs the statement
// for real. RequireColumnScopedGuards holds the column-scoped row guards of a
// table to the columns they list in UPDATE OF (PER-R22).
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
	"testing"

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
}

// Node is one plan node. Rows and the removed counts are averages per loop,
// as EXPLAIN reports them.
type Node struct {
	Type             string  `json:"Node Type"`
	Relationship     string  `json:"Parent Relationship"`
	Relation         string  `json:"Relation Name"`
	Index            string  `json:"Index Name"`
	Rows             float64 `json:"Actual Rows"`
	Loops            float64 `json:"Actual Loops"`
	RemovedByFilter  float64 `json:"Rows Removed by Filter"`
	RemovedByRecheck float64 `json:"Rows Removed by Index Recheck"`
	SharedHit        int64   `json:"Shared Hit Blocks"`
	SharedRead       int64   `json:"Shared Read Blocks"`
	Children         []Node  `json:"Plans"`
}

// tableScans are the node types that read rows of a table.
var tableScans = map[string]bool{
	"Seq Scan": true, "Sample Scan": true, "Index Scan": true, "Index Only Scan": true,
	"Bitmap Heap Scan": true, "Tid Scan": true, "Tid Range Scan": true,
}

// Explain runs sql with args under EXPLAIN (ANALYZE, BUFFERS) in a savepoint
// or transaction of db that it rolls back, and returns what PostgreSQL
// reported. The statement really runs, triggers and constraint checks
// included, so it fails as the statement itself would.
func Explain(ctx context.Context, db Beginner, sql string, args ...any) (Analysis, error) {
	savepoint, err := db.Begin(ctx)
	if err != nil {
		return Analysis{}, err
	}
	var output []byte
	err = savepoint.QueryRow(ctx,
		`EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, SUMMARY OFF, FORMAT JSON) `+sql, args...).Scan(&output)
	if rollbackErr := savepoint.Rollback(ctx); err == nil {
		err = rollbackErr
	}
	if err != nil {
		return Analysis{}, err
	}
	return parse(output)
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

// Examined returns the rows of relation the plan's table scans read over all
// their loops: the rows they returned and the rows their filters and index
// rechecks removed. Index entries that a scan skipped without reading a row,
// as a bitmap index scan or a whole-index walk does, count in Blocks only.
func (n Node) Examined(relation string) int64 {
	var examined float64
	n.walk(func(node Node, _ int) {
		if node.Relation == relation && tableScans[node.Type] {
			examined += (node.Rows + node.RemovedByFilter + node.RemovedByRecheck) * node.Loops
		}
	})
	return int64(examined + 0.5)
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
// index, rows over all loops, rows removed and shared blocks, for test logs.
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
// rolled back; batches and copies are not probed.
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
	p.fired = map[Trigger]int64{}
}

func (p *Probe) analyze(ctx context.Context, sql string, args ...any) {
	p.t.Helper()
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
