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
// child Audit before Run cancellation advances. The Audit controller remains
// responsible for collection, hold release, child-Run deletion, and purge.
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
    SELECT audit.audit_id, audit.state
      FROM audits AS audit JOIN live_project USING (project_id)
     WHERE audit.deletion_requested_at IS NULL
     ORDER BY audit.created_at, audit.audit_id
     FOR UPDATE OF audit SKIP LOCKED
     LIMIT 1
), changed AS (
    UPDATE audits AS audit
       SET state = CASE WHEN candidate.state IN ('draft', 'completed', 'cancelled', 'failed')
                        THEN 'deleting' ELSE 'cancelling' END,
           dispatch_state = 'closed',
           deletion_requested_at = clock_timestamp(),
           stop_reason_code = 'project_deleting',
           stop_reason_message = 'The owning Project is being deleted.',
           revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM candidate
     WHERE audit.audit_id = candidate.audit_id
    RETURNING audit.audit_id, audit.revision, audit.next_event_sequence, audit.state
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, entity_revision, summary)
    SELECT audit_id, next_event_sequence - 1, 'audit.delete_requested', audit_id, revision,
           jsonb_build_object('state', state, 'source', 'project-deletion')
      FROM changed
)
SELECT audit_id FROM changed`, claim.ProjectID, claim.OwnerID, claim.ClaimID, claim.Phase).Scan(&auditID)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	return err == nil, err
}
