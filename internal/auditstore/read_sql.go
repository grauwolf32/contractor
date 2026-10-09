package auditstore

// SQL statements for read.go.

// listItemAttemptsSQL lists the execution attempts of items $3 in owner $1's
// Audit $2, each with its execution's role, terminal outcome, Run provenance
// and Run-deleted flag, ordered by item and attempt. $4 is one more than the
// projection bound so the caller can detect overflow.
// Used by PostgresStore.ListItemAttempts.
var listItemAttemptsSQL = `
SELECT member.execution_item_id, member.execution_id, member.item_id,
       member.item_attempt, execution.role, member.state,
       member.collection_disposition, member.result_ref, member.result_digest,
       execution.terminal_outcome, execution.run_id, execution.run_provenance,
       execution.run_deleted_at IS NOT NULL AS run_deleted,
       member.created_at, member.collected_at
  FROM audit_execution_items AS member
  JOIN audit_executions AS execution USING (execution_id, audit_id)
  JOIN audits AS audit USING (audit_id)
 WHERE audit.owner_id = $1 AND audit.audit_id = $2
   AND member.item_id = ANY($3::text[])
 ORDER BY member.item_id, member.item_attempt, member.execution_item_id
 LIMIT $4`

// listCoverageSQL pages the coverage of round $2's items in Audit $1 together
// with each item's ordinal and task, keyset-paginated by item ordinal after $3
// and limited to $4 rows. Used by PostgresStore.ListCoverage.
var listCoverageSQL = `
SELECT item.audit_id, item.round_id, item.item_id, item.ordinal,
       item.item_key, item.subject_key, item.coverage_status,
       item.coverage_requested, item.coverage_completed, item.coverage_gaps,
       item.coverage_rationale, item.coverage_result_ref, item.coverage_result_digest,
       item.coverage_updated_at, item.task_ref, item.task_digest
  FROM audit_items AS item
 WHERE item.audit_id = $1 AND item.round_id = $2
   AND item.ordinal > $3
 ORDER BY item.ordinal, item.item_id LIMIT $4`

// listRoleReceiptsSQL lists the collection receipts of Audit $1's discovery and
// assessment executions in round $2 (NULL matches executions without a round),
// ordered by role and attempt. $3 is one more than the per-round bound so the
// caller can detect overflow. Used by PostgresStore.listRoleReceipts.
var listRoleReceiptsSQL = `
SELECT execution.collection_receipt_id, execution.audit_id, execution.execution_id, execution.run_id,
       execution.terminal_outcome, execution.terminal_run_generation,
       execution.terminal_run_sequence, execution.collection_disposition,
       execution.collection_error_code, execution.collection_request_digest, execution.collected_at
  FROM audit_executions AS execution
 WHERE execution.audit_id = $1 AND (execution.role = 'prepare' OR (
       execution.role IN ('discovery', 'assessment') AND execution.round_id IS NOT DISTINCT FROM $2))
   AND execution.collection_receipt_id IS NOT NULL
 ORDER BY execution.role, execution.workflow_role, execution.role_attempt, execution.execution_id
 LIMIT $3`

// listCollectionReceiptsSQL lists the collection receipts of Audit $1's
// collected executions, newest first. $2 is one more than the reconcile bound
// so the caller can detect overflow. Used by PostgresStore.ReconcileSnapshot.
var listCollectionReceiptsSQL = `
SELECT collection_receipt_id, audit_id, execution_id, run_id,
       terminal_outcome, terminal_run_generation, terminal_run_sequence,
       collection_disposition, collection_error_code, collection_request_digest, collected_at
  FROM audit_executions
 WHERE audit_id = $1 AND collection_receipt_id IS NOT NULL
 ORDER BY collected_at DESC, collection_receipt_id DESC
 LIMIT $2`
