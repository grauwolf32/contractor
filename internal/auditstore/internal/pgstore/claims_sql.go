package pgstore

// SQL statements for claims.go.

// reconcilableAuditStatesSQL lists the Audit states the controller
// reconciles: every auditstore.AuditState but draft and the terminal ones
// (TestReconcilableAuditStatesSQLMatchesDeclaration), which is also the
// predicate of audits_reconcile_idx. The claim names the whole list rather
// than the states its predicate tells apart, so the planner proves that the
// partial index holds every candidate and has no other index condition on
// state to combine with it. Used by claimAuditSQL.
const reconcilableAuditStatesSQL = `('active', 'waiting_review', 'paused', 'finalizing', 'cancelling', 'deleting')`

// claimAuditSQL leases up to $3 reconcilable Audits to holder $1 for $2 ms:
// active, finalizing, cancelling or deleting; paused with a submitted or
// collecting execution; or waiting_review past the deadline with no report, or
// with an expired report review or item approval lacking a live replacement.
// Free or expired claim rows are taken FOR UPDATE SKIP LOCKED in the order
// (epoch, updated_at, audit_id) and leased through the shared claim helper,
// which advances their epoch and returns the new claims. Used by
// ClaimStore.Claim.
//
// The candidates come from audits_reconcile_idx, and each candidate's claim
// is read by its primary key, so a pass reads the reconcilable Audits and
// their claims, never the claim rows that every retained draft and terminal
// Audit keeps until it is purged. The two OFFSET 0 lookups keep the claim
// reads parameterized by the candidate instead of letting the planner hash
// the whole claim table. The materialized ordered stream feeds the locking
// lookup row by row, so LIMIT stops locking after the batch, rows skipped as
// locked included. The CASE applies the paused and waiting_review conditions
// after the state list has chosen the index. OFFSET 0 also keeps the
// live-authority probe correlated to an exact item, rather than expanding its
// volatile view into repeated scans of all pending reviews.
var claimAuditSQL = `
WITH reconcilable AS (
    SELECT audit.audit_id, audit.updated_at
      FROM audits AS audit
     WHERE audit.state IN ` + reconcilableAuditStatesSQL + `
       AND CASE audit.state
               WHEN 'paused' THEN EXISTS (
                   SELECT 1 FROM audit_executions AS execution
                    WHERE execution.audit_id = audit.audit_id
                      AND execution.state IN ('submitted', 'collecting')
               )
               WHEN 'waiting_review' THEN (
                   (
                       audit.deadline_at <= clock_timestamp()
                       AND NOT EXISTS (
                           SELECT 1 FROM audit_review_requests AS report
                            WHERE report.audit_id = audit.audit_id
                              AND report.subject_kind = 'audit-report'
                       )
                   )
                   OR EXISTS (
                       SELECT 1
                         FROM audit_review_requests AS review
                        WHERE review.audit_id = audit.audit_id
                          AND review.subject_kind = 'audit-report'
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
                          AND ` + acceptedItemSQL("item") + `
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
                              SELECT 1 FROM audit_live_item_reviews AS live
                               WHERE live.audit_id=item.audit_id AND live.item_id=item.item_id
                              OFFSET 0
                          )
                   )
               )
               ELSE true
           END
), ordered AS MATERIALIZED (
    SELECT reconcilable.audit_id
      FROM reconcilable
      JOIN LATERAL (
          SELECT claim.epoch
            FROM audit_controller_claims AS claim
           WHERE claim.audit_id = reconcilable.audit_id
             AND ` + auditClaims.AvailableCondition("claim") + `
          OFFSET 0
      ) AS available ON true
     ORDER BY available.epoch, reconcilable.updated_at, reconcilable.audit_id
), candidates AS (
    SELECT locked.claim_tid
      FROM ordered
      JOIN LATERAL (
          SELECT claim.ctid AS claim_tid
            FROM audit_controller_claims AS claim
           WHERE claim.audit_id = ordered.audit_id
             AND ` + auditClaims.AvailableCondition("claim") + `
             FOR UPDATE SKIP LOCKED
      ) AS locked ON true
     LIMIT $3
)` + auditClaims.LeaseSQL()
