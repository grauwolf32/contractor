package pgstore

// executionTaskOriginSQL uses the existing unique (execution_id,batch_ordinal)
// key and Task primary key. The second-member probe distinguishes a single
// assignment from a batch without counting/hydrating the remaining members.
// Used by QueryStore.ExecutionTaskOrigin during authenticated finding intake.
var executionTaskOriginSQL = `
SELECT task.document
FROM audit_execution_items AS member
JOIN audit_executions AS execution USING (execution_id)
JOIN audit_tasks AS task ON task.task_id=member.task_id AND task.audit_id=member.audit_id
WHERE member.execution_id=$1 AND execution.run_id=$2
  AND member.batch_ordinal=$3 AND execution.role='task'
  AND (NOT $4 OR NOT EXISTS (
      SELECT 1 FROM audit_execution_items AS other
      WHERE other.execution_id=$1 AND other.batch_ordinal=1))`
