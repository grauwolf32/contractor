package pgstore

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

// PurgeProjectFindings releases one bounded slice before intake receipts and
// ordinary Project Artifacts. Audits are already gone. Clearing duplicate
// pointers is cleanup of a deleting owner, never an analyst operation.
func (s *OwnerTxStore) PurgeProjectFindings(ctx context.Context, projectID string) (bool, error) {
	var deleting bool
	if err := s.tx.QueryRow(ctx, "SELECT lifecycle_state='deleting' FROM projects WHERE project_id=$1", projectID).Scan(&deleting); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return false, auditstore.ErrNotFound
		}
		return false, err
	}
	if !deleting {
		return false, auditstore.ErrPrecondition
	}
	for _, query := range projectFindingPurgeSQL {
		tag, err := s.tx.Exec(ctx, query, projectID)
		if err != nil {
			return false, err
		}
		if tag.RowsAffected() != 0 {
			return false, nil
		}
	}
	return true, nil
}
