package auditstore

import (
	"context"
	"errors"
	"fmt"
)

// Claim leases a bounded set of Audits with reconcilable work. A paused Audit
// only observes and collects the Runs it already submitted, so it is claimed
// only while such an execution exists; cancel and delete move it out of
// paused and make it claimable again.
func (s *PostgresStore) Claim(ctx context.Context, params ClaimParams) ([]ControllerClaim, error) {
	if err := validateClaimParams(params); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
WITH candidates AS (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
      JOIN audits AS audit USING (audit_id)
     WHERE (
               audit.state IN ('active', 'finalizing', 'cancelling', 'deleting')
               OR (
                   audit.state = 'paused'
                   AND EXISTS (
                       SELECT 1 FROM audit_executions AS execution
                        WHERE execution.audit_id = audit.audit_id
                          AND execution.state IN ('submitted', 'collecting')
                   )
               )
               OR (
                   audit.state = 'waiting_review'
                   AND (
                       (
                           audit.deadline_at <= clock_timestamp()
                           AND NOT EXISTS (
                               SELECT 1 FROM audit_report_candidates AS report
                                WHERE report.audit_id = audit.audit_id
                           )
                       )
                       OR EXISTS (
                           SELECT 1
                             FROM audit_report_candidates AS report
                             JOIN audit_review_requests AS review
                               ON review.request_id = report.request_id
                              AND review.audit_id = report.audit_id
                            WHERE report.audit_id = audit.audit_id
                              AND (
                                  review.state = 'expired'
                                  OR (review.state = 'pending' AND review.expires_at <= clock_timestamp())
                              )
                       )
                       OR EXISTS (
                           SELECT 1
                             FROM audit_items AS item
                             JOIN audit_review_requests AS review
                               ON review.audit_id=item.audit_id
                              AND review.subject_kind='audit-item-action'
                              AND review.subject_id=item.item_id
                              AND review.kind=item.approval_kind
                              AND review.subject_revision=1
                              AND review.subject_digest=item.approval_subject_digest
                            WHERE item.audit_id=audit.audit_id
                              AND item.state IN ('ready','awaiting_review')
                              AND item.approval_kind <> 'none'
                              AND (review.state='expired' OR (
                                  review.state IN ('pending','decided')
                                  AND review.expires_at <= clock_timestamp()
                              ))
                              AND (review.state <> 'decided' OR EXISTS (
                                  SELECT 1 FROM audit_review_decisions AS decision
                                   WHERE decision.request_id=review.request_id
                                     AND decision.action='approve'
                              ))
                              AND NOT EXISTS (
                                  SELECT 1 FROM audit_review_requests AS live
                                   WHERE live.audit_id=item.audit_id
                                     AND live.subject_kind='audit-item-action'
                                     AND live.subject_id=item.item_id
                                     AND live.kind=item.approval_kind
                                     AND live.subject_revision=1
                                     AND live.subject_digest=item.approval_subject_digest
                                     AND (live.expires_at IS NULL OR live.expires_at > clock_timestamp())
                                     AND (live.state='pending' OR (live.state='decided' AND EXISTS (
                                         SELECT 1 FROM audit_review_decisions AS decision
                                          WHERE decision.request_id=live.request_id
                                            AND decision.action='approve'
                                     )))
                              )
                       )
                   )
               )
           )
       AND (claim.holder_id IS NULL OR claim.expires_at <= clock_timestamp())
     ORDER BY claim.epoch, audit.updated_at, audit.audit_id
     FOR UPDATE OF claim SKIP LOCKED
     LIMIT $3
), claimed AS (
    UPDATE audit_controller_claims AS claim
       SET epoch = epoch + 1,
           holder_id = $1,
           claimed_at = clock_timestamp(),
           expires_at = clock_timestamp() + $2::bigint * interval '1 millisecond'
      FROM candidates
     WHERE claim.audit_id = candidates.audit_id
    RETURNING claim.audit_id, claim.holder_id, claim.epoch,
              claim.claimed_at, claim.expires_at
)
SELECT audit_id, holder_id, epoch, claimed_at, expires_at
  FROM claimed
 ORDER BY claimed_at, audit_id`, params.HolderID, params.Lease.Milliseconds(), params.Limit)
	if err != nil {
		return nil, fmt.Errorf("claim Audits: %w", err)
	}
	defer rows.Close()
	result := make([]ControllerClaim, 0, params.Limit)
	for rows.Next() {
		claim, scanErr := scanClaim(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit claim: %w", scanErr)
		}
		result = append(result, claim)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit claims: %w", err)
	}
	return result, nil
}

func (s *PostgresStore) ReleaseClaim(ctx context.Context, claim ControllerClaim) error {
	if err := validateClaimIdentity(claim); err != nil {
		return err
	}
	tag, err := s.db.Exec(ctx, `
UPDATE audit_controller_claims
   SET holder_id = NULL, claimed_at = NULL, expires_at = NULL
 WHERE audit_id = $1 AND holder_id = $2 AND epoch = $3`,
		claim.AuditID, claim.HolderID, claim.Epoch,
	)
	if err != nil {
		return fmt.Errorf("release Audit claim: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return ErrClaimLost
	}
	return nil
}

func scanClaim(row scanner) (ControllerClaim, error) {
	var result ControllerClaim
	var epoch int64
	if err := row.Scan(
		&result.AuditID, &result.HolderID, &epoch,
		&result.ClaimedAt, &result.ExpiresAt,
	); err != nil {
		return ControllerClaim{}, err
	}
	if epoch <= 0 || !result.ExpiresAt.After(result.ClaimedAt) {
		return ControllerClaim{}, errors.New("stored Audit claim is invalid")
	}
	result.Epoch = uint64(epoch)
	return result, nil
}
