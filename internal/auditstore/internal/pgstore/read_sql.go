package pgstore

// SQL statements for read.go.

// listReadyItemsSQL returns the first $3 ready items of each Workflow role in
// Round $2 of Audit $1, in item order. A role belongs to one lane class, so a
// class whose dispatch window is full cannot hide the ready items of the other
// class (AUX-R3), and within a role the lowest ordinals come first (AUX-R13).
// ready_roles walks the distinct roles through audit_items_ready_role_idx, one
// index probe per role; at most LIM-ASC-2 roles, so at most 16 * $3 rows.
// Used by ClaimStore.listReadyItems.
var listReadyItemsSQL = `
WITH RECURSIVE ready_roles(workflow_role) AS (
    (SELECT item.workflow_role
       FROM audit_items AS item
      WHERE item.audit_id = $1 AND item.round_id = $2 AND item.state = 'ready'
      ORDER BY item.workflow_role
      LIMIT 1)
    UNION ALL
    SELECT (SELECT item.workflow_role
              FROM audit_items AS item
             WHERE item.audit_id = $1 AND item.round_id = $2 AND item.state = 'ready'
               AND item.workflow_role > ready_roles.workflow_role
             ORDER BY item.workflow_role
             LIMIT 1)
      FROM ready_roles
     WHERE ready_roles.workflow_role IS NOT NULL
)
SELECT ` + prefixedItemColumns("item") + `
  FROM ready_roles
 CROSS JOIN LATERAL (
       SELECT role_item.*
         FROM audit_items AS role_item
        WHERE role_item.audit_id = $1 AND role_item.round_id = $2 AND role_item.state = 'ready'
          AND role_item.workflow_role = ready_roles.workflow_role
        ORDER BY role_item.ordinal, role_item.item_id
        LIMIT $3) AS item
 ORDER BY item.ordinal, item.item_id`

// listItemsPageSQL returns one page of the items of owner $1's Audit $2 in
// round ordinal, item ordinal and item ID order. It filters before the
// keyset: only round $3, state $4 and subject key $5, each when it is not
// NULL. ($6, $7, $8) is the (round ordinal, item ordinal, item ID) cursor of
// the previous page; a NULL $6 starts at the first page. At most $9 rows.
// Used by QueryStore.ListItemsPage.
var listItemsPageSQL = `
SELECT ` + prefixedItemColumns("item") + `
  FROM audit_items AS item
  JOIN audit_rounds AS round USING (round_id, audit_id)
  JOIN audits AS audit USING (audit_id)
 WHERE audit.owner_id = $1 AND item.audit_id = $2
   AND ($3::text IS NULL OR item.round_id = $3)
   AND ($4::text IS NULL OR item.state = $4)
   AND ($5::text IS NULL OR item.subject_key = $5)
   AND ($6::integer IS NULL OR (round.ordinal, item.ordinal, item.item_id) > ($6, $7, $8))
 ORDER BY round.ordinal, item.ordinal, item.item_id
 LIMIT $9`

// listItemAttemptsSQL lists the execution attempts of items $3 in owner $1's
// Audit $2, each with its execution's role, terminal outcome, Run provenance
// and Run-deleted flag, ordered by item and attempt. $4 is one more than the
// projection bound so the caller can detect overflow.
// Used by QueryStore.ListItemAttempts.
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
// and limited to $4 rows. Used by QueryStore.ListCoverage.
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
// caller can detect overflow. Used by ClaimStore.listRoleReceipts.
var listRoleReceiptsSQL = `
SELECT ` + receiptSummaryColumns("execution") + `
  FROM audit_executions AS execution
 WHERE execution.audit_id = $1 AND (execution.role = 'prepare' OR (
       execution.role IN ('discovery', 'assessment') AND execution.round_id IS NOT DISTINCT FROM $2))
   AND execution.collection_receipt_id IS NOT NULL
 ORDER BY execution.role, execution.workflow_role, execution.role_attempt, execution.execution_id
 LIMIT $3`

// listCollectionReceiptsSQL lists the collection receipts of Audit $1's
// collected executions, newest first. $2 is one more than the reconcile bound
// so the caller can detect overflow. Used by ClaimStore.GetReconcileSnapshot.
var listCollectionReceiptsSQL = `
SELECT ` + receiptSummaryColumns("audit_executions") + `
  FROM audit_executions
 WHERE audit_id = $1 AND collection_receipt_id IS NOT NULL
 ORDER BY collected_at DESC, collection_receipt_id DESC
 LIMIT $2`

// nextItemAttemptsSQL returns, for each of items $4 that is ready in Audit $1
// under the live controller claim ($1-$3), one more than its highest recorded
// execution attempt. An item that is not ready has no row. The unique
// (item_id, item_attempt) index serves each maximum.
// Used by ClaimStore.NextItemAttempts.
var nextItemAttemptsSQL = `
SELECT item.item_id, COALESCE(max(member.item_attempt), 0) + 1
  FROM audit_items AS item
  JOIN audit_controller_claims AS claim USING (audit_id)
  LEFT JOIN audit_execution_items AS member ON member.item_id = item.item_id
 WHERE item.audit_id = $1 AND ` + auditClaims.LiveCondition("claim", "$2", "$3") + `
   AND item.item_id = ANY($4::text[]) AND item.state = 'ready'
 GROUP BY item.item_id`
