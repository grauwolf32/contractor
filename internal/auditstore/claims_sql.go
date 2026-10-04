package auditstore

// SQL statements for claims.go.

// claimAuditSQL leases up to $3 reconcilable Audits to holder $1 for $2 ms:
// active, finalizing, cancelling or deleting; paused with a submitted or
// collecting execution; or waiting_review past the deadline with no report, or
// with an expired report review or item approval lacking a live replacement.
// Free or expired claim rows are taken FOR UPDATE SKIP LOCKED, their epoch is
// bumped, and the new claims are returned. Used by PostgresStore.Claim.
var claimAuditSQL = `
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
 ORDER BY claimed_at, audit_id`
