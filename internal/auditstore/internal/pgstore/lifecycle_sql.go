package pgstore

// SQL statements for lifecycle.go.

// settledUndispatchedEvents record how many items dispatch closed and one
// review.expired event per expired item review, in request order.
var settledUndispatchedEvents = auditEvents(
	auditEvent("'items.dispatch_closed'", "audit.audit_id", "NULL",
		"jsonb_build_object('count', (SELECT count(*) FROM changed_items))"),
	auditEventsFrom("expired_reviews AS review", "review.request_id",
		"'review.expired'", "review.request_id", "review.revision",
		`jsonb_build_object(
               'subjectKind', 'audit-item-action', 'subjectId', review.subject_id,
               'kind', review.kind
           )`),
)

// settleUndispatchedSQL settles up to $4 pending, awaiting_review or ready
// items (SKIP LOCKED) of a finalizing, cancelling or deleting Audit under the
// live controller claim ($1-$3), records a coverage gap, expires their pending
// reviews and records items.dispatch_closed plus one review.expired each.
// Returns whether the Audit qualified and how many items were settled.
// Used by ClaimStore.SettleUndispatched.
var settleUndispatchedSQL = `
WITH live_claim AS MATERIALIZED (
    ` + auditClaims.LiveClaimSQL("$1", "$2", "$3") + `
), target_audit AS MATERIALIZED (
    SELECT audit.audit_id, audit.state
      FROM audits AS audit JOIN live_claim USING (audit_id)
     WHERE audit.state IN ('finalizing', 'cancelling', 'deleting')
     FOR UPDATE OF audit
), locked_items AS MATERIALIZED (
    SELECT item.item_id, item.last_execution_item_id IS NOT NULL AS had_attempt
      FROM audit_items AS item JOIN target_audit USING (audit_id)
     WHERE item.state IN ('pending', 'awaiting_review', 'ready')
     ORDER BY item.round_id, item.ordinal, item.item_id
     FOR UPDATE OF item SKIP LOCKED
     LIMIT $4
), candidates AS MATERIALIZED (
    SELECT locked.item_id, locked.had_attempt, target_audit.state,
           CASE
               WHEN locked.had_attempt AND target_audit.state = 'finalizing'
                   THEN 'audit-closed-before-retry'
               WHEN locked.had_attempt THEN 'audit-cancelled-before-retry'
               WHEN target_audit.state = 'finalizing' THEN 'audit-closed-before-dispatch'
               ELSE 'audit-cancelled-before-dispatch'
           END AS gap,
           CASE
               WHEN locked.had_attempt AND target_audit.state = 'finalizing'
                   THEN 'Audit dispatch closed before a retry of this item was submitted.'
               WHEN locked.had_attempt
                   THEN 'Audit cancellation closed this item before a retry was submitted.'
               WHEN target_audit.state = 'finalizing'
                   THEN 'Audit dispatch closed before this item was submitted.'
               ELSE 'Audit cancellation closed this item before dispatch.'
           END AS message
      FROM locked_items AS locked CROSS JOIN target_audit
), changed_items AS (
    -- Settling an item records the closure gap on its coverage in the same
    -- row update.
    UPDATE audit_items AS item
       SET state = 'settled',
           final_disposition = CASE WHEN candidates.state = 'finalizing'
               THEN 'excluded' ELSE 'execution-cancelled' END,
           updated_at = GREATEST(clock_timestamp(), item.updated_at + interval '1 microsecond'),
           coverage_status = CASE WHEN candidates.had_attempt THEN item.coverage_status
               WHEN candidates.state = 'finalizing' THEN 'not-tested' ELSE 'blocked' END,
           coverage_gaps = CASE
               WHEN item.coverage_gaps ? candidates.gap
               THEN item.coverage_gaps
               ELSE item.coverage_gaps || jsonb_build_array(candidates.gap)
           END,
           coverage_rationale = CASE
               WHEN NOT candidates.had_attempt THEN candidates.message
               WHEN octet_length(concat_ws(' ', nullif(item.coverage_rationale, ''), candidates.message)) <= 4096
                   THEN concat_ws(' ', nullif(item.coverage_rationale, ''), candidates.message)
               ELSE item.coverage_rationale
           END,
           coverage_updated_at = GREATEST(clock_timestamp(), item.coverage_updated_at + interval '1 microsecond')
      FROM candidates
     WHERE item.item_id = candidates.item_id
    RETURNING item.item_id, item.audit_id
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
), advanced_audit AS (
    UPDATE audits AS audit
       SET ` + settledUndispatchedEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM target_audit
     WHERE audit.audit_id = target_audit.audit_id
       AND EXISTS (SELECT 1 FROM changed_items)
    RETURNING audit.audit_id, ` + settledUndispatchedEvents.appended() + `
)
SELECT EXISTS(SELECT 1 FROM target_audit), count(*) FROM changed_items`

// dispatchHoldReleasedEvents record a released dispatch hold.
var dispatchHoldReleasedEvents = auditEvents(auditEvent(
	"'audit.dispatch_hold_released'", "audit.audit_id", "audit.revision", "'{}'::jsonb",
))

// releaseDispatchHoldSQL releases the dispatch hold of the Audit under the live
// controller claim ($1-$3) once dispatch is closed, the hold is held and no
// execution is still an intent, recording audit.dispatch_hold_released. The
// caller completes the trailing SELECT with the Audit columns.
// Used by ClaimStore.ReleaseDispatchHold.
var releaseDispatchHoldSQL = `
WITH live_claim AS MATERIALIZED (
    ` + auditClaims.LiveClaimSQL("$1", "$2", "$3") + `
), changed AS (
    UPDATE audits AS audit
       SET hold_state = 'released',
           ` + dispatchHoldReleasedEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM live_claim
     WHERE audit.audit_id = live_claim.audit_id
       AND audit.dispatch_state = 'closed' AND audit.hold_state = 'held'
       AND NOT EXISTS (
           SELECT 1 FROM audit_executions AS execution
            WHERE execution.audit_id = audit.audit_id AND execution.state = 'intent'
       )
    RETURNING audit.*, ` + dispatchHoldReleasedEvents.appended() + `
)
SELECT `

// nextLiveRunForDeletionSQL returns the child Run of the earliest execution
// that still exists in workflow_runs, for deleting Audit $1 held by live claim
// ($2, $3). No row means no child Run is left or the gate failed.
// Used by ClaimStore.NextLiveRunForDeletion.
var nextLiveRunForDeletionSQL = `
SELECT run.run_id
  FROM audits AS audit
  JOIN audit_controller_claims AS claim USING (audit_id)
  JOIN audit_executions AS execution USING (audit_id)
  JOIN workflow_runs AS run ON run.run_id = execution.run_id
 WHERE audit.audit_id = $1 AND audit.state = 'deleting'
   AND ` + auditClaims.LiveCondition("claim", "$2", "$3") + `
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
   AND ` + auditClaims.LiveCondition("claim", "$2", "$3") + `
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
