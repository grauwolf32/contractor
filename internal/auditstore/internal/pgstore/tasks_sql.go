package pgstore

import "github.com/grauwolf32/contractor/internal/auditstore"

// Per-Workflow sparse index probes bound candidate work before sorting and
// hydration. Only small index keys cross the at-most-16 alias probes.
const readyTasksSQL = `
WITH candidates AS MATERIALIZED (
 SELECT candidate.* FROM unnest($2::text[]) AS allowed(alias)
 CROSS JOIN LATERAL (
  SELECT task_id,ordinal,metadata_bytes,attention FROM audit_tasks
   WHERE audit_id=$1 AND workflow_alias=allowed.alias AND state='ready'
   ORDER BY attention DESC,ordinal LIMIT 64
 ) AS candidate
), selected AS MATERIALIZED (
 SELECT * FROM candidates ORDER BY attention DESC,ordinal LIMIT 64
), bounded AS (
 SELECT task_id,ordinal,sum(metadata_bytes) OVER (ORDER BY attention DESC,ordinal) AS used_bytes
 FROM selected
)
SELECT ` + taskColumnSQL + ` FROM audit_tasks JOIN bounded USING(task_id,ordinal)
 WHERE used_bytes<=1048576 ORDER BY attention DESC,ordinal`

var settledTaskEvents = auditEvents(auditEvent("'tasks.dispatch_closed'", "audit.audit_id", "audit.revision", "jsonb_build_object('count',(SELECT count(*) FROM changed_tasks))"))

var settleUndispatchedTasksSQL = `
WITH live_claim AS MATERIALIZED (
 ` + auditClaims.LiveClaimSQL("$1", "$2", "$3") + `
), target_audit AS MATERIALIZED (
 SELECT audit.audit_id,audit.state FROM audits AS audit JOIN live_claim USING(audit_id)
  WHERE audit.state IN ('finalizing','cancelling','deleting') FOR UPDATE OF audit
), candidates AS MATERIALIZED (
 SELECT task.task_id FROM audit_tasks AS task JOIN target_audit USING(audit_id)
  WHERE task.state='ready' ORDER BY task.ordinal FOR UPDATE OF task LIMIT $4
), changed_tasks AS (
 UPDATE audit_tasks AS task SET state='settled',updated_at=clock_timestamp(),
  gap_code=CASE WHEN target_audit.state='finalizing' THEN 'audit-closed-before-dispatch' ELSE 'audit-cancelled-before-dispatch' END
 FROM candidates,target_audit WHERE task.task_id=candidates.task_id
 RETURNING task.task_id
), changed_audit AS (
 UPDATE audits AS audit SET updated_at=clock_timestamp(),` + settledTaskEvents.advance() + `
 FROM target_audit WHERE audit.audit_id=target_audit.audit_id AND EXISTS(SELECT 1 FROM changed_tasks)
 RETURNING ` + settledTaskEvents.appended() + `
)
SELECT EXISTS(SELECT 1 FROM target_audit),(SELECT count(*) FROM changed_tasks)
 FROM (SELECT count(*) FROM changed_audit) AS emitted`

// Closed state literals let generic PostgreSQL plans use the sparse state's
// statistics. A generic state parameter can prefer the ordinal index and
// filter the entire settled history before finding a few ready tasks.
func taskStateFilterSQL(state *auditstore.TaskState) string {
	if state == nil {
		return ""
	}
	switch *state {
	case auditstore.TaskReady:
		return "AND state='ready'"
	case auditstore.TaskSelected:
		return "AND state='selected'"
	case auditstore.TaskSettled:
		return "AND state='settled'"
	default:
		return "AND false"
	}
}

// SQL for the common accepted-task owner. The candidate LIMIT runs before the
// metadata window, so a page never examines the containing Audit history.
const taskColumnSQL = `document,source_key,source,ordinal,iteration,batch_digest,
state,attempts,last_execution_id,result,gap_code,created_at,updated_at,hypothesis_snapshot,attention,finding_result,stop_record`

const taskPageSQL = `
WITH candidates AS MATERIALIZED (
    SELECT document,source_key,source,ordinal,iteration,batch_digest,state,attempts,
           last_execution_id,result,gap_code,created_at,updated_at,hypothesis_snapshot,attention,finding_result,stop_record,
           metadata_bytes+coalesce(octet_length(stop_record::text),0) AS metadata_bytes
      FROM audit_tasks
     WHERE audit_id=$1 AND ordinal>$2 %s
     ORDER BY ordinal
     LIMIT $3
), bounded AS (
    SELECT candidates.*,sum(metadata_bytes) OVER (ORDER BY ordinal) AS used_bytes,
           count(*) OVER () AS candidate_count
      FROM candidates
)
SELECT ` + taskColumnSQL + `,candidate_count FROM bounded
 WHERE used_bytes <= 1048576
 ORDER BY ordinal`

const tasksBySourceSQL = `SELECT ` + taskColumnSQL + ` FROM audit_tasks
 WHERE audit_id=$1 AND source_key=ANY($2::text[]) ORDER BY ordinal`

const getTaskSQL = `SELECT ` + taskColumnSQL + ` FROM audit_tasks
 WHERE audit_id=$1 AND task_id=$2`

// insertTasksSQL inserts one already validated bounded array. Natural source
// identities are checked under the locked Audit row before this statement.
const insertTasksSQL = `
INSERT INTO audit_tasks
 (task_id,audit_id,source_key,source,document,batch_digest,ordinal,iteration,hypothesis_snapshot,attention)
SELECT entry->'document'->>'task_id',$1,entry->>'source_key',entry->'source',
       entry->'document',entry->>'batch_digest',(entry->>'ordinal')::integer,
       (entry->>'iteration')::integer,entry->'hypothesis_snapshot',(entry->>'attention')::boolean
  FROM json_array_elements($2::json) AS entry`

var tasksAcceptedEvents = auditEvents(auditEvent("'tasks.accepted'", "audit.audit_id", "audit.revision", "jsonb_build_object('createdCount',$5::integer)"))

var advanceAcceptedTasksSQL = `
UPDATE audits AS audit
   SET task_count=$2,task_iteration=$3,iteration_task_count=$4,
       initial_work_offset=COALESCE($6::integer,initial_work_offset),
       updated_at=clock_timestamp(),` + tasksAcceptedEvents.advance() + `
 WHERE audit_id=$1
RETURNING audit.revision,` + tasksAcceptedEvents.appended()

var preparedSourceEvents = auditEvents(auditEvent("'execution.work_available'", "$2::text", "NULL", "jsonb_build_object('workCount',$4::integer)"))

var recordPreparedSourceSQL = `
WITH source AS (
 UPDATE audit_executions SET prepared_work_source=$3::json,prepared_work_count=$4,
        updated_at=clock_timestamp()
  WHERE audit_id=$1 AND execution_id=$2 AND prepared_work_source IS NULL
 RETURNING audit_id
)
UPDATE audits AS audit SET updated_at=clock_timestamp(),` + preparedSourceEvents.advance() + `
 FROM source WHERE audit.audit_id=source.audit_id
 RETURNING ` + preparedSourceEvents.appended()
