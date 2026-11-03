package pgstore

// SQL statements for lifecycle.go.

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

// firstLiveChildRunSQL is the child Run of the earliest execution of the
// Audit aliased audit whose Run still exists in workflow_runs, as a LATERAL
// subquery. An execution whose Run is stamped deleted is skipped through
// audit_executions_audit_run_cleanup_idx: run_deleted_at IS NULL selects the
// index range of the unstamped executions, and run_deleted_at leads the
// ORDER BY because an IS NULL condition does not make a column constant for
// the planner, so only that order lets the index deliver (created_at,
// execution_id) and the probe stop at the first live Run instead of reading
// every execution of the Audit and every Run (V414-029). The stamp is set in
// the transaction that deletes the Run (internal/rundeletion, the only Run
// delete), and the deferred contractor_require_matching_audit_run check
// rejects a stamp while the Run row exists; the EXISTS probe by primary key
// keeps the meaning "the Run still exists" for an execution whose Run was
// removed without its stamp. Used by nextLiveRunForDeletionSQL and
// purgeClaimedAuditSQL.
const firstLiveChildRunSQL = `(
      SELECT execution.run_id
        FROM audit_executions AS execution
       WHERE execution.audit_id = audit.audit_id
         AND execution.run_id IS NOT NULL AND execution.run_deleted_at IS NULL
         AND EXISTS (SELECT 1 FROM workflow_runs AS run WHERE run.run_id = execution.run_id)
       ORDER BY execution.run_deleted_at, execution.created_at, execution.execution_id
       LIMIT 1
  )`

// nextLiveRunForDeletionSQL returns the child Run of the earliest execution
// that still exists in workflow_runs, for deleting Audit $1 held by live claim
// ($2, $3). No row means no child Run is left or the gate failed.
// Used by ClaimStore.NextLiveRunForDeletion.
var nextLiveRunForDeletionSQL = `
SELECT next.run_id
  FROM audits AS audit
  JOIN audit_controller_claims AS claim USING (audit_id)
  CROSS JOIN LATERAL ` + firstLiveChildRunSQL + ` AS next
 WHERE audit.audit_id = $1 AND audit.state = 'deleting'
   AND ` + auditClaims.LiveCondition("claim", "$2", "$3")

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
   AND NOT EXISTS (SELECT 1 FROM audit_tasks AS task WHERE task.audit_id=audit.audit_id AND task.state<>'settled')
   AND NOT ` + uncollectedExecutionSQL + `
   AND NOT EXISTS (SELECT 1 FROM ` + firstLiveChildRunSQL + ` AS live)
 FOR UPDATE OF audit, claim`
