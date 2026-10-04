package auditstore

// SQL statements for lifecycle.go.

// settleUndispatchedSQL settles up to $4 pending, awaiting_review or ready
// items (SKIP LOCKED) of a finalizing, cancelling or deleting Audit under the
// live controller claim ($1-$3), records a coverage gap, expires their pending
// reviews and appends items.dispatch_closed plus one review.expired each.
// Returns whether the Audit qualified and how many items were settled.
// Used by PostgresStore.SettleUndispatched.
var settleUndispatchedSQL = `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), target_audit AS MATERIALIZED (
    SELECT audit.audit_id, audit.state
      FROM audits AS audit JOIN live_claim USING (audit_id)
     WHERE audit.state IN ('finalizing', 'cancelling', 'deleting')
     FOR UPDATE OF audit
), candidates AS MATERIALIZED (
    SELECT item.item_id
      FROM audit_items AS item JOIN target_audit USING (audit_id)
     WHERE item.state IN ('pending', 'awaiting_review', 'ready')
     ORDER BY item.round_id, item.ordinal, item.item_id
     FOR UPDATE OF item SKIP LOCKED
     LIMIT $4
), changed_items AS (
    UPDATE audit_items AS item
       SET state = 'settled',
           final_disposition = CASE WHEN target_audit.state = 'finalizing'
               THEN 'excluded' ELSE 'execution-cancelled' END,
           updated_at = GREATEST(clock_timestamp(), item.updated_at + interval '1 microsecond')
      FROM candidates, target_audit
     WHERE item.item_id = candidates.item_id
    RETURNING item.item_id, item.audit_id,
              item.last_execution_item_id IS NOT NULL AS had_attempt
), settlement AS (
    SELECT changed_items.item_id, changed_items.had_attempt, target_audit.state,
           CASE
               WHEN changed_items.had_attempt AND target_audit.state = 'finalizing'
                   THEN 'audit-closed-before-retry'
               WHEN changed_items.had_attempt THEN 'audit-cancelled-before-retry'
               WHEN target_audit.state = 'finalizing' THEN 'audit-closed-before-dispatch'
               ELSE 'audit-cancelled-before-dispatch'
           END AS gap,
           CASE
               WHEN changed_items.had_attempt AND target_audit.state = 'finalizing'
                   THEN 'Audit dispatch closed before a retry of this item was submitted.'
               WHEN changed_items.had_attempt
                   THEN 'Audit cancellation closed this item before a retry was submitted.'
               WHEN target_audit.state = 'finalizing'
                   THEN 'Audit dispatch closed before this item was submitted.'
               ELSE 'Audit cancellation closed this item before dispatch.'
           END AS message
      FROM changed_items CROSS JOIN target_audit
), changed_coverage AS (
    UPDATE audit_coverage_rows AS coverage
       SET status = CASE WHEN settlement.had_attempt THEN coverage.status
               WHEN settlement.state = 'finalizing' THEN 'not-tested' ELSE 'blocked' END,
           gaps = CASE
               WHEN coverage.gaps ? settlement.gap
               THEN coverage.gaps
               ELSE coverage.gaps || jsonb_build_array(settlement.gap)
           END,
           rationale = CASE
               WHEN NOT settlement.had_attempt THEN settlement.message
               WHEN octet_length(concat_ws(' ', nullif(coverage.rationale, ''), settlement.message)) <= 4096
                   THEN concat_ws(' ', nullif(coverage.rationale, ''), settlement.message)
               ELSE coverage.rationale
           END,
           updated_at = GREATEST(clock_timestamp(), coverage.updated_at + interval '1 microsecond')
      FROM settlement
     WHERE coverage.item_id = settlement.item_id
), expired_reviews AS (
    UPDATE audit_review_requests AS review
       SET state = 'expired', revision = review.revision + 1,
           updated_at = GREATEST(clock_timestamp(), review.updated_at + interval '1 microsecond')
      FROM changed_items AS item
     WHERE review.audit_id = item.audit_id
       AND review.subject_kind = 'audit-item-action'
       AND review.subject_id = item.item_id AND review.state = 'pending'
    RETURNING review.request_id, review.audit_id, review.subject_id,
              review.kind, review.revision
), expired_review_count AS MATERIALIZED (
    SELECT count(*)::bigint AS expired FROM expired_reviews
), advanced_audit AS (
    UPDATE audits AS audit
       SET revision = audit.revision + 1 + expired_review_count.expired,
           next_event_sequence = audit.next_event_sequence + 1 + expired_review_count.expired,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM target_audit, expired_review_count
     WHERE audit.audit_id = target_audit.audit_id
       AND EXISTS (SELECT 1 FROM changed_items)
    RETURNING audit.audit_id,
              audit.next_event_sequence - 1 - expired_review_count.expired AS dispatch_sequence
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, summary)
    SELECT advanced_audit.audit_id, advanced_audit.dispatch_sequence,
           'items.dispatch_closed', advanced_audit.audit_id,
           jsonb_build_object('count', (SELECT count(*) FROM changed_items))
      FROM advanced_audit
), expired_review_events AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT review.audit_id,
           advanced_audit.dispatch_sequence + row_number() OVER (ORDER BY review.request_id),
           'review.expired', review.request_id, review.revision,
           jsonb_build_object(
               'subjectKind', 'audit-item-action', 'subjectId', review.subject_id,
               'kind', review.kind
           )
      FROM expired_reviews AS review JOIN advanced_audit USING (audit_id)
)
SELECT EXISTS(SELECT 1 FROM target_audit), count(*) FROM changed_items`

// releaseDispatchHoldSQL releases the dispatch hold of the Audit under the live
// controller claim ($1-$3) once dispatch is closed, the hold is held and no
// execution is still an intent, appending audit.dispatch_hold_released. The
// caller completes the trailing SELECT with the Audit columns.
// Used by PostgresStore.ReleaseDispatchHold.
var releaseDispatchHoldSQL = `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), changed AS (
    UPDATE audits AS audit
       SET hold_state = 'released', revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM live_claim
     WHERE audit.audit_id = live_claim.audit_id
       AND audit.dispatch_state = 'closed' AND audit.hold_state = 'held'
       AND NOT EXISTS (
           SELECT 1 FROM audit_executions AS execution
            WHERE execution.audit_id = audit.audit_id AND execution.state = 'intent'
       )
    RETURNING audit.*
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, entity_revision, summary)
    SELECT audit_id, next_event_sequence - 1, 'audit.dispatch_hold_released', audit_id, revision, '{}'::jsonb
      FROM changed
)
SELECT `

// nextLiveRunForDeletionSQL returns the child Run of the earliest execution
// that still exists in workflow_runs, for deleting Audit $1 held by live claim
// ($2, $3). No row means no child Run is left or the gate failed.
// Used by PostgresStore.NextLiveRunForDeletion.
var nextLiveRunForDeletionSQL = `
SELECT run.run_id
  FROM audits AS audit
  JOIN audit_controller_claims AS claim USING (audit_id)
  JOIN audit_executions AS execution USING (audit_id)
  JOIN workflow_runs AS run ON run.run_id = execution.run_id
 WHERE audit.audit_id = $1 AND audit.state = 'deleting'
   AND claim.holder_id = $2 AND claim.epoch = $3
   AND claim.expires_at > clock_timestamp()
 ORDER BY execution.created_at, execution.execution_id
 LIMIT 1`

// purgeClaimedAuditSQL locks a drained deleting Audit and its live claim
// ($1-$3) for purge and returns its Project ID. Drained means dispatch closed,
// hold not held, deletion requested, every item settled, every execution
// collected and no child Run left in workflow_runs.
// Used by purgeClaimedAudit.
var purgeClaimedAuditSQL = `
SELECT audit.project_id
  FROM audits AS audit
  JOIN audit_controller_claims AS claim USING (audit_id)
 WHERE audit.audit_id = $1 AND audit.state = 'deleting'
   AND audit.dispatch_state = 'closed' AND audit.hold_state <> 'held'
   AND audit.deletion_requested_at IS NOT NULL
   AND claim.holder_id = $2 AND claim.epoch = $3
   AND claim.expires_at > clock_timestamp()
   AND NOT EXISTS (
       SELECT 1 FROM audit_items AS item
        WHERE item.audit_id = audit.audit_id AND item.state <> 'settled'
   )
   AND NOT EXISTS (
       SELECT 1 FROM audit_executions AS execution
        WHERE execution.audit_id = audit.audit_id AND execution.state <> 'collected'
   )
   AND NOT EXISTS (
       SELECT 1
         FROM audit_executions AS execution
         JOIN workflow_runs AS run ON run.run_id = execution.run_id
        WHERE execution.audit_id = audit.audit_id
   )
 FOR UPDATE OF audit, claim`
