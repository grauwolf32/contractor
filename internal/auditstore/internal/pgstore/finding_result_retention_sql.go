package pgstore

// findingResultTasksSQL reads only the accepted members in this collection.
// These exact task refs and source facts are retained independently before the
// same transaction commits its ordinary collection receipt.
const findingResultTasksSQL = `
SELECT member.execution_item_id,item.task_ref,item.task_digest,item.origin,execution.run_id,execution.run_provenance
FROM audit_execution_items AS member
JOIN audit_items AS item ON item.item_id=member.item_id AND item.audit_id=member.audit_id
JOIN audit_executions AS execution ON execution.execution_id=member.execution_id
WHERE member.audit_id=$1 AND member.execution_id=$2 AND member.execution_item_id=ANY($3::text[])
ORDER BY member.execution_item_id`
