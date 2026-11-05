package auditstoretest

// testItemsSQL is a test-only inspection of historical Round fixtures.
const testItemsSQL = `
SELECT item.item_id,item.audit_id,item.round_id,item.item_key,item.ordinal,item.kind,
       item.subject_key,item.task_ref,item.task_digest,item.origin,item.workflow_role,
       item.state,item.final_disposition,item.accepted_result_ref,item.accepted_result_digest,
       item.last_execution_item_id,item.created_at,item.updated_at,item.retained_dependencies
 FROM audit_items AS item JOIN audit_rounds AS round USING(audit_id,round_id)
 WHERE item.audit_id=$1 AND round.state<>'proposed'
 ORDER BY round.ordinal,item.ordinal,item.item_id`
