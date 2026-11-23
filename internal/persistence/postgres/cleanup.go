package postgres

import (
	"context"
	"errors"
	"fmt"
)

// CleanupBatchRows is the shared row bound for physical lifecycle batches.
const CleanupBatchRows = 1000

// CleanupBudget is the row and byte budget of one bounded cleanup pass, a
// transaction that deletes in one or more statements and commits before the
// next pass starts. It is the one definition of the rule "at most RowLimit
// row mutations and ByteLimit bytes per pass, and the first unit always
// goes", which the Eval purge (LIM-EVM-20) and the Project purge of finding
// proposal receipts (LIM-DEL-3) apply. Rows and Bytes are what the pass has
// charged so far.
//
// What a unit costs is the caller's to state, not measured here: a constant
// count of the row mutations that deleting one row makes, its own and those
// of its triggers, and an SQL expression of the bytes it is charged, usually
// the stored size of the columns it holds (pg_column_size, which counts a
// compressed or out-of-line value at its stored size) plus any constant the
// caller adds for rows its triggers rewrite. A pass that has charged nothing
// is fresh: its first unit is taken even when it alone exceeds ByteLimit, so
// a nonempty remainder always makes progress.
//
// The Audit purge does not use it. LIM-REV-10 bounds an Audit purge pass by
// rows alone, which its statements count down without measuring any row; it
// adopts this budget when an Audit pass needs a byte bound.
type CleanupBudget struct {
	RowLimit, ByteLimit int64
	Rows, Bytes         int64
}

// Fits reports whether a unit of rows row mutations and bytes charged bytes
// still fits the pass. Its row cost must always fit; its charge must fit too
// unless the pass is still fresh.
func (b *CleanupBudget) Fits(rows, bytes int64) bool {
	return b.Rows+rows <= b.RowLimit && (b.Rows == 0 || b.Bytes+bytes <= b.ByteLimit)
}

// Charge records a unit the pass has deleted or rewritten.
func (b *CleanupBudget) Charge(rows, bytes int64) {
	b.Rows += rows
	b.Bytes += bytes
}

// BoundedDelete names the rows that one bounded delete statement takes and
// what each costs. Table and Filter select the rows; Filter binds Args from
// $5. Order is the SQL order in which they are taken, which an index should
// serve; empty takes them in whatever order the plan finds them. RowBytes is
// the SQL expression of the bytes one row is charged and RowCost the row
// mutations deleting it makes, both stated by the caller as CleanupBudget
// describes.
type BoundedDelete struct {
	Table, Filter, Order, RowBytes string
	RowCost                        int64
	Args                           []any
}

// DeleteWithinBudget deletes, in one statement, the longest prefix of the
// rows d names whose cost fits the budget, and charges it. At most
// RowLimit+1 rows are read, and each is charged before any is deleted. It
// reports whether no matching row is left: false when the budget stopped the
// statement, also when it had no room for a single row.
func DeleteWithinBudget(ctx context.Context, db DBTX, budget *CleanupBudget, d BoundedDelete) (bool, error) {
	if budget == nil || d.Table == "" || d.RowBytes == "" || d.Filter == "" || d.RowCost < 1 ||
		budget.RowLimit < 1 || budget.RowLimit > CleanupBatchRows || budget.ByteLimit < 1 {
		return false, errors.New("bounded cleanup arguments are invalid")
	}
	rows := budget.RowLimit - budget.Rows
	if rows < d.RowCost {
		return false, nil
	}
	candidate := fmt.Sprintf(unorderedCandidateSQL, d.Table, d.RowBytes, d.Filter)
	if d.Order != "" {
		candidate = fmt.Sprintf(orderedCandidateSQL, d.Table, d.RowBytes, d.Filter, d.Order)
	}
	params := append([]any{rows, budget.ByteLimit - budget.Bytes, budget.Rows == 0, d.RowCost}, d.Args...)
	var deleted, bytes, read int64
	err := db.QueryRow(ctx, fmt.Sprintf(boundedDeleteSQL, d.Table, candidate), params...).Scan(&deleted, &bytes, &read)
	if err != nil {
		return false, err
	}
	budget.Charge(deleted*d.RowCost, bytes)
	return deleted == read, nil
}
