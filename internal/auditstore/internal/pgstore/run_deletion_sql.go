package pgstore

// SQL statements for run_deletion.go.

// lockRunDeletionAuditsSQL locks FOR UPDATE, in audit_id order, owner $1's
// Audits linked to Run $2 through an AuditExecution, a finding proposal
// receipt or a proposal Audit hold, and returns their IDs.
// Used by RunDeletionTx.LockRunDeletionAudits.
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

// collectedExecutionPredicate is the gate an Audit-managed Run passes before
// deletion: its AuditExecution (aliased execution) is 'collected' with Run
// provenance and a collection receipt.
const collectedExecutionPredicate = `
execution.state = 'collected'
   AND execution.run_provenance IS NOT NULL
   AND execution.collection_receipt_id IS NOT NULL`

// markExecutionRunDeletedSQL stamps run_deleted_at (first value kept) on
// AuditExecution $1 of Run $2 when it passes collectedExecutionPredicate, and
// strictly advances updated_at. Used by RunDeletion.MarkExecutionRunDeleted.
var markExecutionRunDeletedSQL = `
UPDATE audit_executions AS execution
   SET run_deleted_at = COALESCE(execution.run_deleted_at, clock_timestamp()),
       updated_at = GREATEST(clock_timestamp(), execution.updated_at + interval '1 microsecond')
 WHERE execution.execution_id = $1 AND execution.run_id = $2
   AND ` + collectedExecutionPredicate

// collectedExecutionRunsSQL returns the IDs among AuditExecutions $1 whose
// Run is the matching element of $2 and that pass
// collectedExecutionPredicate. Used by RunDeletion.CollectedExecutionRuns.
var collectedExecutionRunsSQL = `
SELECT execution.execution_id
  FROM audit_executions AS execution
  JOIN unnest($1::text[], $2::text[]) AS wanted(execution_id, run_id)
    ON wanted.execution_id = execution.execution_id AND wanted.run_id = execution.run_id
 WHERE ` + collectedExecutionPredicate
