package postgres

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"
)

// CleanupBatchRows is the shared row bound for physical lifecycle batches.
const CleanupBatchRows = 1000

// DeleteAuditReceiptsBatch removes at most limit receipts after the caller
// has locked a drained deleting Audit. The one creation receipt stays until
// the Audit DELETE, whose Eval tombstone trigger reads it. Other mutation
// families are untouched.
func DeleteAuditReceiptsBatch(ctx context.Context, tx pgx.Tx, auditID string, limit int) (int, error) {
	if tx == nil || auditID == "" || limit < 1 || limit > CleanupBatchRows {
		return 0, errors.New("Audit receipt purge arguments are invalid")
	}
	tag, err := tx.Exec(ctx, `WITH batch AS (
  SELECT ctid FROM mutation_receipts WHERE audit_id = $1 AND operation <> 'audit.create' LIMIT $2
)
DELETE FROM mutation_receipts AS receipt USING batch WHERE receipt.ctid = batch.ctid`, auditID, limit)
	if err != nil {
		return 0, WrapError("purge Audit mutation receipt batch", err)
	}
	return int(tag.RowsAffected()), nil
}
