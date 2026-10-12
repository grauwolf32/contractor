package findingintake

import (
	"context"
	"errors"
	"fmt"
	"strings"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// receiptPurgeBatchSize and receiptPurgeBatchBytes bound one batch of a
// Project receipt purge (LIM-DEL-3): at most this many receipts, whose stored
// report projections and evidence take at most this many bytes. The first
// receipt of a batch goes even when it alone is larger, so every batch makes
// progress.
var (
	receiptPurgeBatchSize        = 1000
	receiptPurgeBatchBytes int64 = 64 << 20
)

// PurgeProjectReceipts deletes the next batch of finding proposal receipts of
// a deleting Project in tx and reports whether none is left. Project deletion
// calls it in each purge transaction after the Project's Runs and Audits, and
// with the Audits every hold, contribution, finding and assessment that
// referenced a receipt, are gone (DEL-R61). It deletes in the Project
// lifecycle purge mode of the Artifact purger; the receipt trigger refuses a
// receipt whose Project is not deleting.
func PurgeProjectReceipts(ctx context.Context, tx pgx.Tx, projectID string) (bool, error) {
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
	var purged, read int64
	if err := tx.QueryRow(
		ctx, purgeProjectReceiptBatchSQL, projectID, receiptPurgeBatchSize, receiptPurgeBatchBytes,
	).Scan(&purged, &read); err != nil {
		return false, fmt.Errorf("purge Project finding proposal receipts: %w", err)
	}
	return purged == read, nil
}
