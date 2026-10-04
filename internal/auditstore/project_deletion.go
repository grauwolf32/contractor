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
	err := s.db.QueryRow(ctx, `
WITH live_project AS MATERIALIZED (
    SELECT project_id
      FROM projects
     WHERE project_id = $1 AND owner_id = $2
       AND lifecycle_state = 'deleting' AND deletion_phase = $4
       AND deletion_claim_id = $3
     FOR UPDATE
), candidate AS MATERIALIZED (
    SELECT audit.*
      FROM audits AS audit JOIN live_project USING (project_id)
     WHERE audit.deletion_requested_at IS NULL
     ORDER BY audit.created_at, audit.audit_id
     FOR UPDATE OF audit SKIP LOCKED
     LIMIT 1
), `+deletionRequestCTEs(auditDeletionByProject)+`
SELECT audit_id FROM changed`, claim.ProjectID, claim.OwnerID, claim.ClaimID, claim.Phase).Scan(&auditID)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	return err == nil, err
}
