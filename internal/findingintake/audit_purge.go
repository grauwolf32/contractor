package findingintake

import (
	"context"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// DeleteAuditHoldsBatch releases at most limit holds in the caller's drained
// Audit purge transaction. The existing retention trigger updates receipts.
func DeleteAuditHoldsBatch(ctx context.Context, tx pgx.Tx, auditID string, limit int) (int, error) {
	if tx == nil || !contracts.ValidResourceID(auditID) || limit < 1 || limit > persistencepostgres.CleanupBatchRows {
		return 0, ErrInvalid
	}
	tag, err := tx.Exec(ctx, `WITH batch AS (
  SELECT ctid FROM finding_proposal_audit_holds WHERE audit_id = $1 LIMIT $2
)
DELETE FROM finding_proposal_audit_holds AS hold USING batch WHERE hold.ctid = batch.ctid`, auditID, limit)
	if err != nil {
		return 0, persistencepostgres.WrapError("purge Audit finding hold batch", err)
	}
	return int(tag.RowsAffected()), nil
}
