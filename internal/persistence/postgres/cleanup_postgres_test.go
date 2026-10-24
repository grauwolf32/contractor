package postgres

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
)

// DeleteWithinBudget stops at the row limit, at the byte limit, and before a
// unit that the rows left cannot hold; a fresh pass takes its first row even
// when it alone exceeds the byte limit, and a pass that has charged bytes
// never does. With an order it takes the rows in that order. It charges what
// it measured before the deletion and reports whether no matching row is
// left.
func TestPostgresDeleteWithinBudgetBoundsRowsAndBytes(t *testing.T) {
	f := newKeyedFixture(t)
	const receipts = 30
	for i := range receipts {
		key := EvalReceiptKey("owner", "keyed-evaluation", "experiments", "experiment-create", fmt.Sprintf("budget-%02d", i))
		receipt := Receipt{
			Operation: "experiment-create", ResourceID: "experiments", RequestDigest: keyedDigest("1"),
			Response: []byte(`{"pad":"` + strings.Repeat("x", 1000*(i+1)) + `"}`), OwnerID: "owner", ProjectID: "keyed-evaluation",
		}
		if err := claimReceipt(f.ctx, f.pool, key, receipt); err != nil {
			t.Fatal(err)
		}
	}
	var largest, total int64
	if err := f.pool.QueryRow(f.ctx, `SELECT max(`+receiptRowBytes+`)::bigint, sum(`+receiptRowBytes+`)::bigint
FROM mutation_receipts WHERE key_scope[1] = 'eval'`).Scan(&largest, &total); err != nil {
		t.Fatal(err)
	}
	left := func() int64 {
		t.Helper()
		var n int64
		if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM mutation_receipts WHERE key_scope[1] = 'eval'`).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}
	orderedPass := func(budget CleanupBudget, rowCost int64, order string) (CleanupBudget, bool) {
		t.Helper()
		var gone bool
		err := InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			if _, err := tx.Exec(f.ctx, `SELECT set_config('contractor.eval_purge', 'on', true)`); err != nil {
				return err
			}
			var err error
			gone, err = DeleteWithinBudget(f.ctx, tx, &budget, BoundedDelete{
				Table: "mutation_receipts", Filter: `owner_id = $5 AND project_id = $6 AND key_scope[1] = 'eval'`,
				Order: order, RowBytes: receiptRowBytes, RowCost: rowCost, Args: []any{"owner", "keyed-evaluation"},
			})
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
		return budget, gone
	}
	pass := func(budget CleanupBudget, rowCost int64) (CleanupBudget, bool) {
		t.Helper()
		return orderedPass(budget, rowCost, "")
	}
	sizes := func() []int64 {
		t.Helper()
		rows, err := f.pool.Query(f.ctx, `SELECT octet_length(response) FROM mutation_receipts
WHERE key_scope[1] = 'eval' ORDER BY octet_length(response) DESC`)
		if err != nil {
			t.Fatal(err)
		}
		out, err := pgx.CollectRows(rows, pgx.RowTo[int64])
		if err != nil {
			t.Fatal(err)
		}
		return out
	}

	remaining := int64(receipts)
	if got := left(); got != remaining {
		t.Fatalf("Eval receipts = %d, want %d", got, remaining)
	}
	if budget, gone := pass(CleanupBudget{RowLimit: 10, ByteLimit: 1 << 30}, 1); gone || budget.Rows != 10 {
		t.Fatalf("row-bound pass = (%+v, %t)", budget, gone)
	}
	remaining -= 10
	// With an order the statement takes the rows in that order: the two
	// largest receipts, then under a one-byte budget the largest alone.
	before := sizes()
	if budget, gone := orderedPass(CleanupBudget{RowLimit: 2, ByteLimit: 1 << 30}, 1, "octet_length(response) DESC"); gone ||
		budget.Rows != 2 || fmt.Sprint(sizes()) != fmt.Sprint(before[2:]) {
		t.Fatalf("ordered pass = (%+v, %t), left %v of %v", budget, gone, sizes(), before)
	}
	if budget, gone := orderedPass(CleanupBudget{RowLimit: 10, ByteLimit: 1}, 1, "octet_length(response) DESC"); gone ||
		budget.Rows != 1 || fmt.Sprint(sizes()) != fmt.Sprint(before[3:]) {
		t.Fatalf("ordered fresh pass under a one-byte budget = (%+v, %t), left %v of %v", budget, gone, sizes(), before)
	}
	remaining -= 3
	if budget, gone := pass(CleanupBudget{RowLimit: 10, ByteLimit: 1 << 30}, 3); gone || budget.Rows != 9 {
		t.Fatalf("pass of three-row units = (%+v, %t)", budget, gone)
	}
	remaining -= 3
	if budget, gone := pass(CleanupBudget{RowLimit: 10, ByteLimit: 1}, 1); gone || budget.Rows != 1 || budget.Bytes < 1 {
		t.Fatalf("fresh pass under a one-byte budget = (%+v, %t), want its first row", budget, gone)
	}
	remaining--
	if budget, gone := pass(CleanupBudget{RowLimit: 10, ByteLimit: 1 << 20, Rows: 1, Bytes: 1 << 20}, 1); gone || budget.Rows != 1 || budget.Bytes != 1<<20 {
		t.Fatalf("spent pass = (%+v, %t), want nothing more", budget, gone)
	}
	if budget, gone := pass(CleanupBudget{RowLimit: 10, ByteLimit: 1 << 30, Rows: 9}, 2); gone || budget.Rows != 9 {
		t.Fatalf("pass without room for a unit = (%+v, %t), want nothing more", budget, gone)
	}
	if got := left(); got != remaining {
		t.Fatalf("Eval receipts = %d, want %d", got, remaining)
	}
	budget, gone := pass(CleanupBudget{RowLimit: CleanupBatchRows, ByteLimit: 2 * largest}, 1)
	if gone || budget.Bytes > 2*largest || budget.Rows < 1 {
		t.Fatalf("byte-bound pass = (%+v, %t)", budget, gone)
	}
	remaining -= budget.Rows
	var stored int64
	if err := f.pool.QueryRow(f.ctx, `SELECT sum(`+receiptRowBytes+`)::bigint FROM mutation_receipts WHERE key_scope[1] = 'eval'`).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if budget, gone := pass(CleanupBudget{RowLimit: CleanupBatchRows, ByteLimit: total}, 1); !gone || budget.Rows != remaining || budget.Bytes != stored {
		t.Fatalf("last pass = (%+v, %t), want %d rows of %d bytes", budget, gone, remaining, stored)
	}
	if got := left(); got != 0 {
		t.Fatalf("Eval receipts after the last pass = %d", got)
	}
	for _, invalid := range []CleanupBudget{{RowLimit: 0, ByteLimit: 1}, {RowLimit: CleanupBatchRows + 1, ByteLimit: 1}, {RowLimit: 1}} {
		if _, err := DeleteWithinBudget(f.ctx, f.pool, &invalid, BoundedDelete{
			Table: "mutation_receipts", Filter: "true", RowBytes: receiptRowBytes, RowCost: 1,
		}); err == nil {
			t.Fatalf("budget %+v accepted", invalid)
		}
	}
	valid := CleanupBudget{RowLimit: 1, ByteLimit: 1}
	for _, invalid := range []BoundedDelete{
		{Filter: "true", RowBytes: receiptRowBytes, RowCost: 1},
		{Table: "mutation_receipts", RowBytes: receiptRowBytes, RowCost: 1},
		{Table: "mutation_receipts", Filter: "true", RowCost: 1},
		{Table: "mutation_receipts", Filter: "true", RowBytes: receiptRowBytes},
	} {
		if _, err := DeleteWithinBudget(f.ctx, f.pool, &valid, invalid); err == nil {
			t.Fatalf("delete %+v accepted", invalid)
		}
	}
}

// receiptRowBytes measures every column of mutation_receipts, so a new
// column cannot escape the Project purge's byte budget.
func TestPostgresReceiptRowBytesMeasuresEveryColumn(t *testing.T) {
	f := newKeyedFixture(t)
	var columns []string
	if err := f.pool.QueryRow(f.ctx, `
SELECT array_agg(column_name::text)
FROM information_schema.columns
WHERE table_schema = current_schema() AND table_name = 'mutation_receipts'`).Scan(&columns); err != nil {
		t.Fatal(err)
	}
	var measured []string
	for _, match := range regexp.MustCompile(`pg_column_size\((\w+)\)`).FindAllStringSubmatch(receiptRowBytes, -1) {
		measured = append(measured, match[1])
	}
	slices.Sort(columns)
	slices.Sort(measured)
	if strings.Join(columns, ",") != strings.Join(measured, ",") {
		t.Fatalf("mutation_receipts columns %v, receiptRowBytes measures %v", columns, measured)
	}
}
