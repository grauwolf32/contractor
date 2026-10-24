package postgres

import (
	"context"
	"errors"
	"fmt"
)

// CleanupBatchRows is the shared row bound for physical lifecycle batches.
const CleanupBatchRows = 1000

// CleanupBudget is the row and stored-byte budget of one bounded cleanup
// pass, a transaction that deletes in several statements and commits before
// the next pass starts. Rows and Bytes are what the pass has spent so far:
// the rows its statements deleted or rewrote, counting the rows their
// triggers write, and the stored size of those rows (pg_column_size, which
// counts a compressed or out-of-line value at its stored size). A pass that
// has spent nothing is fresh: its first unit is taken even when it alone
// exceeds ByteLimit, so a nonempty remainder always makes progress.
type CleanupBudget struct {
	RowLimit, ByteLimit int64
	Rows, Bytes         int64
}

// Fits reports whether a unit of rows row mutations and bytes stored bytes
// still fits the pass. Its row cost must always fit; its size must fit too
// unless the pass is still fresh.
func (b *CleanupBudget) Fits(rows, bytes int64) bool {
	return b.Rows+rows <= b.RowLimit && (b.Rows == 0 || b.Bytes+bytes <= b.ByteLimit)
}

// Charge records a unit the pass has deleted or rewritten.
func (b *CleanupBudget) Charge(rows, bytes int64) {
	b.Rows += rows
	b.Bytes += bytes
}

// DeleteWithinBudget deletes, in one statement, the longest run of the rows of
// table that match filter whose cost fits the budget, and charges it. rowBytes
// is the SQL expression of one row's stored size, including the rows its
// deletion triggers rewrite, and rowCost the row mutations deleting one row
// makes. filter binds args from $5. The rows are measured before any of them
// is deleted, and at most RowLimit+1 of them are read. It reports whether no
// matching row is left: false when the budget stopped the statement, also
// when it had no room for a single row.
func DeleteWithinBudget(
	ctx context.Context, db DBTX, budget *CleanupBudget, table, rowBytes, filter string, rowCost int64, args ...any,
) (bool, error) {
	if budget == nil || table == "" || rowBytes == "" || filter == "" || rowCost < 1 ||
		budget.RowLimit < 1 || budget.RowLimit > CleanupBatchRows || budget.ByteLimit < 1 {
		return false, errors.New("bounded cleanup arguments are invalid")
	}
	rows := budget.RowLimit - budget.Rows
	if rows < rowCost {
		return false, nil
	}
	params := append([]any{rows, budget.ByteLimit - budget.Bytes, budget.Rows == 0, rowCost}, args...)
	var deleted, bytes, read int64
	err := db.QueryRow(ctx, fmt.Sprintf(boundedDeleteSQL, table, rowBytes, filter), params...).Scan(&deleted, &bytes, &read)
	if err != nil {
		return false, err
	}
	budget.Charge(deleted*rowCost, bytes)
	return deleted == read, nil
}
