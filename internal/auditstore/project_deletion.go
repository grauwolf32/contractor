package auditstore

import (
	"context"
	"errors"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

// RequestProjectOwnedDeletion commits a live Project deletion fence into one
// child Audit before Run cancellation advances, through the same deletion
// request as owner deletion. It first locks the Project under the deletion
// claim through projectstore and changes nothing when the claim is lost. The
// Audit controller remains responsible for collection, hold release,
// child-Run deletion, and purge.
func (s *PostgresStore) RequestProjectOwnedDeletion(
	ctx context.Context, claim projectstore.DeletionClaim,
) (bool, error) {
	var changed bool
	err := persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		changed = false
		err := projectstore.NewPostgresTxStore(tx).LockDeletionClaim(ctx, claim)
		if errors.Is(err, projectstore.ErrDeletionClaimLost) {
			return nil
		}
		if err != nil {
			return err
		}
		var auditID string
		err = tx.QueryRow(ctx, requestProjectOwnedDeletionSQL+deletionRequestCTEs(auditDeletionByProject)+`
SELECT audit_id FROM changed`, claim.ProjectID).Scan(&auditID)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil
		}
		changed = err == nil
		return err
	})
	return changed, err
}
