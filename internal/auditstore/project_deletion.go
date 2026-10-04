package auditstore

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"
)

type ProjectDeletionClaim struct {
	ProjectID string
	OwnerID   string
	ClaimID   string
	Phase     string
}

// RequestProjectOwnedDeletion commits a live Project deletion fence into one
// child Audit before Run cancellation advances, through the same deletion
// request as owner deletion. The Audit controller remains responsible for
// collection, hold release, child-Run deletion, and purge.
func (s *PostgresStore) RequestProjectOwnedDeletion(
	ctx context.Context, claim ProjectDeletionClaim,
) (bool, error) {
	var auditID string
	err := s.db.QueryRow(ctx, requestProjectOwnedDeletionSQL+deletionRequestCTEs(auditDeletionByProject)+`
SELECT audit_id FROM changed`, claim.ProjectID, claim.OwnerID, claim.ClaimID, claim.Phase).Scan(&auditID)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	return err == nil, err
}
