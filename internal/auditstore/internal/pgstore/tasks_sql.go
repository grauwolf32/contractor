package pgstore

import "github.com/grauwolf32/contractor/internal/auditstore"

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
state,attempts,last_execution_id,result,gap_code,created_at,updated_at`

const taskPageSQL = `
WITH candidates AS MATERIALIZED (
    SELECT document,source_key,source,ordinal,iteration,batch_digest,state,attempts,
           last_execution_id,result,gap_code,created_at,updated_at,metadata_bytes
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
 (task_id,audit_id,source_key,source,document,batch_digest,ordinal,iteration)
SELECT entry->'document'->>'task_id',$1,entry->>'source_key',entry->'source',
       entry->'document',entry->>'batch_digest',(entry->>'ordinal')::integer,
       (entry->>'iteration')::integer
  FROM json_array_elements($2::json) AS entry`

var tasksAcceptedEvents = auditEvents(auditEvent("'audit.tasks.accepted'", "audit.audit_id", "audit.revision", "jsonb_build_object('createdCount',$5::integer)"))

var advanceAcceptedTasksSQL = `
UPDATE audits AS audit
   SET task_count=$2,task_iteration=$3,iteration_task_count=$4,
       initial_work_offset=COALESCE($6::integer,initial_work_offset),
       updated_at=clock_timestamp(),` + tasksAcceptedEvents.advance() + `
 WHERE audit_id=$1
RETURNING audit.revision,` + tasksAcceptedEvents.appended()
