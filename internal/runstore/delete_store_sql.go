package runstore

// SQL statements for delete_store.go.

// runAuditCollectedSQL reports whether AuditExecution $1 of Run $2 is
// 'collected' with Run provenance and a matching collection receipt, i.e.
// whether the Audit-managed Run may be deleted.
// Used by deleteReleasedTerminalRun.
var runAuditCollectedSQL = `
SELECT EXISTS (
    SELECT 1
      FROM audit_executions AS execution
      JOIN audit_collection_receipts AS receipt
        ON receipt.execution_id = execution.execution_id
       AND receipt.audit_id = execution.audit_id
       AND receipt.run_id = execution.run_id
     WHERE execution.execution_id = $1 AND execution.run_id = $2
       AND execution.state = 'collected' AND execution.run_provenance IS NOT NULL
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
   AND EXISTS (
       SELECT 1 FROM audit_collection_receipts AS receipt
        WHERE receipt.execution_id = audit_executions.execution_id
          AND receipt.audit_id = audit_executions.audit_id
          AND receipt.run_id = audit_executions.run_id
   )`

// markProposalSourceRunDeletedSQL tombstones the retention record of every
// finding proposal received from Run $1: it sets source_run_deleted_at and
// moves the state to 'audit-held' when an Audit hold exists, otherwise to
// 'discarded' with discarded_at. Existing timestamps are kept.
// Used by deleteReleasedTerminalRun.
var markProposalSourceRunDeletedSQL = `
UPDATE finding_proposal_retention AS retention
   SET source_run_deleted_at = COALESCE(retention.source_run_deleted_at, clock_timestamp()),
       state = CASE WHEN EXISTS (
           SELECT 1 FROM finding_proposal_audit_holds AS hold
            WHERE hold.receipt_id = retention.receipt_id
       ) THEN 'audit-held' ELSE 'discarded' END,
       discarded_at = CASE WHEN EXISTS (
           SELECT 1 FROM finding_proposal_audit_holds AS hold
            WHERE hold.receipt_id = retention.receipt_id
       ) THEN NULL ELSE COALESCE(retention.discarded_at, clock_timestamp()) END,
       updated_at = clock_timestamp()
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.receipt_id = retention.receipt_id AND receipt.run_id = $1`

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
             JOIN audit_collection_receipts AS receipt
               ON receipt.execution_id = audit_execution.execution_id
              AND receipt.audit_id = audit_execution.audit_id
              AND receipt.run_id = run.run_id
            WHERE audit_execution.execution_id = run.audit_execution_id
              AND audit_execution.run_id = run.run_id
              AND audit_execution.state = 'collected'
              AND audit_execution.run_provenance IS NOT NULL
       )
  FROM workflow_runs AS run
 WHERE run.run_id = $1 AND run.owner_id = $2`
