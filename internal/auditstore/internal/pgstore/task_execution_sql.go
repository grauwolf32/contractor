package pgstore

// A reservation selects at most one already byte-bounded batch. Scheduler
// settings, Project, claim and Audit are locked before reading its tasks.
const taskSelectionSizeSQL = `SELECT count(*),COALESCE(sum(metadata_bytes),0)
 FROM audit_tasks WHERE audit_id=$1 AND task_id=ANY($2::text[])`

const selectedTasksSQL = `SELECT ` + taskColumnSQL + ` FROM audit_tasks
 WHERE audit_id=$1 AND task_id=ANY($2::text[]) ORDER BY ordinal FOR UPDATE`

const insertTaskExecutionSQL = `INSERT INTO audit_executions
 (execution_id,audit_id,workflow_role,submission_key,request_digest,lane_class)
 VALUES($1,$2,$3,$4,$5,$6) RETURNING `

// The existing execution membership owns attempt associations. A Task's
// document remains in audit_tasks; no duplicate package/manifest is authored.
const insertTaskMembersSQL = `
INSERT INTO audit_execution_items
 (execution_item_id,execution_id,audit_id,task_id,batch_ordinal,item_attempt)
SELECT entry->>'id',$1,$2,entry->>'task_id',(entry->>'ordinal')::integer,
       (entry->>'attempt')::integer
  FROM json_array_elements($3::json) AS entry`

const selectTaskBatchSQL = `UPDATE audit_tasks
 SET state='selected',attempts=attempts+1,last_execution_id=$3,updated_at=clock_timestamp()
 WHERE audit_id=$1 AND task_id=ANY($2::text[]) AND state='ready'`

var taskIntentEvents = auditEvents(auditEvent("'execution.intent_created'", "$2::text", "NULL", "jsonb_build_object('taskCount',$4::integer,'workflow',$5::text)"))

var reserveTaskRunSQL = `
UPDATE audits AS audit
 SET reserved_run_count=reserved_run_count+1,outstanding_run_count=outstanding_run_count+1,
     outstanding_tool_run_count=outstanding_tool_run_count+CASE WHEN $3='tool' THEN 1 ELSE 0 END,
     updated_at=clock_timestamp(),` + taskIntentEvents.advance() + `
 WHERE audit_id=$1 AND state='active' AND dispatch_state='open'
   AND (deadline_at IS NULL OR deadline_at>clock_timestamp())
   AND reserved_run_count<max_submitted_runs
 RETURNING ` + taskIntentEvents.appended()

const taskExecutionMembersSQL = `SELECT task_id FROM audit_execution_items
 WHERE audit_id=$1 AND execution_id=$2
 ORDER BY batch_ordinal LIMIT 65`
