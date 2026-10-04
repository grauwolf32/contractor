package runstore

// SQL statements for delete_store.go.

// runAuditCollectedSQL reports whether AuditExecution $1 of Run $2 is
// 'collected' with Run provenance and a collection receipt, i.e. whether the
// Audit-managed Run may be deleted.
// Used by deleteReleasedTerminalRun.
var runAuditCollectedSQL = `
SELECT EXISTS (
    SELECT 1
      FROM audit_executions AS execution
     WHERE execution.execution_id = $1 AND execution.run_id = $2
       AND execution.state = 'collected' AND execution.run_provenance IS NOT NULL
       AND execution.collection_receipt_id IS NOT NULL
)`

// markAuditRunDeletedSQL stamps run_deleted_at (first value kept) on the
// collected AuditExecution $1 of Run $2 and strictly advances updated_at,
// under the same collected-with-receipt gate as runAuditCollectedSQL. The
// caller requires one affected row. Used by deleteReleasedTerminalRun.
var markAuditRunDeletedSQL = `
UPDATE audit_executions
   SET run_deleted_at = COALESCE(run_deleted_at, clock_timestamp()),
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE execution_id = $1 AND run_id = $2 AND state = 'collected'
   AND run_provenance IS NOT NULL
   AND collection_receipt_id IS NOT NULL`

// markProposalSourceRunDeletedSQL tombstones the retention state of every
// finding proposal received from Run $1: it sets source_run_deleted_at and
// moves the state to 'audit-held' when an Audit hold exists, otherwise to
// 'discarded' with discarded_at. Existing timestamps are kept.
// Used by deleteReleasedTerminalRun.
var markProposalSourceRunDeletedSQL = `
UPDATE finding_proposal_receipts AS receipt
   SET source_run_deleted_at = COALESCE(receipt.source_run_deleted_at, clock_timestamp()),
       retention_state = CASE WHEN EXISTS (
           SELECT 1 FROM finding_proposal_audit_holds AS hold
            WHERE hold.receipt_id = receipt.receipt_id
       ) THEN 'audit-held' ELSE 'discarded' END,
       discarded_at = CASE WHEN EXISTS (
           SELECT 1 FROM finding_proposal_audit_holds AS hold
            WHERE hold.receipt_id = receipt.receipt_id
       ) THEN NULL ELSE COALESCE(receipt.discarded_at, clock_timestamp()) END,
       retention_updated_at = clock_timestamp()
 WHERE receipt.run_id = $1`

// lockRunDeletionAuditsSQL locks FOR UPDATE, in audit_id order, owner $1's
// Audits linked to Run $2 through an AuditExecution, a finding proposal
// receipt or a proposal Audit hold, and returns their IDs.
// Used by lockRunDeletionAudits.
var lockRunDeletionAuditsSQL = `
SELECT audit.audit_id
  FROM audits AS audit
 WHERE audit.owner_id = $1 AND audit.audit_id IN (
       SELECT execution.audit_id FROM audit_executions AS execution WHERE execution.run_id = $2
       UNION
       SELECT receipt.audit_id FROM finding_proposal_receipts AS receipt WHERE receipt.run_id = $2
       UNION
       SELECT hold.audit_id
         FROM finding_proposal_audit_holds AS hold
         JOIN finding_proposal_receipts AS receipt USING (receipt_id)
        WHERE receipt.run_id = $2
   )
 ORDER BY audit.audit_id
 FOR UPDATE OF audit`

// runDeletionBlockerSQL reads, without locking, the deletion gates of Run $1
// owned by $2: its state, whether any Stage allocation is unreleased, and
// whether an Audit-managed Run still lacks a collected AuditExecution with a
// receipt. Used by PostgresStore.RunDeletionBlocker.
var runDeletionBlockerSQL = `
SELECT run.state,
       EXISTS (
           SELECT 1
             FROM stage_executions AS execution
             JOIN stage_allocations AS allocation
               ON allocation.stage_execution_id = execution.stage_execution_id
            WHERE execution.run_id = run.run_id
              AND allocation.release_completed_at IS NULL
       ),
       run.publication_mode = 'audit-managed' AND NOT EXISTS (
           SELECT 1
             FROM audit_executions AS audit_execution
            WHERE audit_execution.execution_id = run.audit_execution_id
              AND audit_execution.run_id = run.run_id
              AND audit_execution.state = 'collected'
              AND audit_execution.run_provenance IS NOT NULL
              AND audit_execution.collection_receipt_id IS NOT NULL
       )
  FROM workflow_runs AS run
 WHERE run.run_id = $1 AND run.owner_id = $2`
