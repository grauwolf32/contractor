package findingintake

import (
	"context"
	"errors"
	"fmt"
	"strings"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// receiptPurgeBatchSize and receiptPurgeBatchBytes bound one batch of a
// Project receipt purge (LIM-DEL-3): at most this many receipts, whose stored
// report projections and evidence take at most this many bytes. The batch is
// a cleanup pass of persistencepostgres.CleanupBudget, so its first receipt
// goes even when it alone is larger, and every batch makes progress.
var (
	receiptPurgeBatchSize  int64 = persistencepostgres.CleanupBatchRows
	receiptPurgeBatchBytes int64 = 64 << 20
)

// receiptPurgeBytes is what one receipt is charged in a Project receipt
// purge: the stored size of its report projection and evidence, which
// pg_column_size reads without detoasting.
const receiptPurgeBytes = `pg_column_size(report_projection) + pg_column_size(evidence)`

// PurgeProjectReceipts deletes the next batch of finding proposal receipts of
// a deleting Project in tx, in receipt_id order, which
// finding_proposal_receipts_project_idx serves, and reports whether none is
// left. Project deletion calls it in each purge transaction after the
// Project's Runs and Audits, and with the Audits every hold, contribution,
// finding and assessment that referenced a receipt, are gone (DEL-R61). It
// deletes in the Project lifecycle purge mode of the Artifact purger; the
// receipt trigger refuses a receipt whose Project is not deleting.
func PurgeProjectReceipts(ctx context.Context, tx pgx.Tx, projectID string, storage *artifactpostgres.Storage) (bool, error) {
	if tx == nil {
		return false, errors.New("PostgreSQL transaction is required")
	}
	if strings.TrimSpace(projectID) == "" {
		return false, ErrInvalid
	}
	ctx, cancel := persistencepostgres.WithCleanupBudget(ctx)
	defer cancel()
	if err := persistencepostgres.ApplyTransactionBudget(ctx, tx); err != nil {
		return false, err
	}
	if _, err := tx.Exec(
		ctx, `SELECT set_config('contractor.lifecycle_purge', 'project', true)`,
	); err != nil {
		return false, fmt.Errorf("enable Project lifecycle purge: %w", err)
	}
	findingsGone, err := ownerstore.NewTx(tx, storage).PurgeProjectFindings(ctx, projectID)
	if err != nil {
		return false, err
	}
	if !findingsGone {
		return false, nil
	}
	budget := persistencepostgres.CleanupBudget{RowLimit: receiptPurgeBatchSize, ByteLimit: receiptPurgeBatchBytes}
	gone, err := persistencepostgres.DeleteWithinBudget(ctx, tx, &budget, persistencepostgres.BoundedDelete{
		Table: "finding_proposal_receipts", Filter: "project_id = $5", Order: "receipt_id",
		RowBytes: receiptPurgeBytes, RowCost: 1, Args: []any{projectID},
	})
	if err != nil {
		return false, fmt.Errorf("purge Project finding proposal receipts: %w", err)
	}
	return gone, nil
}
