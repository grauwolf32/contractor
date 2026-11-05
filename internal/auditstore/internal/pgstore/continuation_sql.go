package pgstore

// Per-workflow reverse seek avoids rescanning settled history to recover the
// last frozen source boundary. Generator outputs never become new sources.
const lastProducerTaskSQL = `SELECT source FROM audit_tasks
 WHERE audit_id=$1 AND workflow_alias=$2 AND source->>'kind'='continuation' ORDER BY ordinal DESC LIMIT 1`

// Read a fixed ordinal window, including gaps. Its consumed boundary is pinned
// on the ordinary continuation Task; each source is visited once per decision.
const continuationTasksSQL = `
WITH candidates AS MATERIALIZED (
 SELECT candidate.* FROM unnest($3::text[]) AS allowed(alias)
 CROSS JOIN LATERAL (
  SELECT task_id,ordinal FROM audit_tasks
   WHERE audit_id=$1 AND workflow_alias=allowed.alias AND ordinal>$2
     AND state='settled' AND source->>'kind'<>'continuation'
   ORDER BY ordinal LIMIT $4
 ) candidate
), selected AS MATERIALIZED (
 SELECT * FROM candidates ORDER BY ordinal LIMIT $4
)
SELECT source.* FROM selected CROSS JOIN LATERAL (
 SELECT task_id,ordinal,hypothesis_snapshot,result,gap_code FROM audit_tasks
 WHERE task_id=selected.task_id AND ordinal=selected.ordinal AND state='settled'
 LIMIT 1
) AS source ORDER BY source.ordinal`

// GetRunCreationIntent resolves only IDs durably selected on its accepted Task.
const taskObservationsSQL = `SELECT task_id,ordinal,hypothesis_snapshot,result,gap_code
 FROM audit_tasks WHERE audit_id=$1 AND task_id=ANY($2::text[])
 AND state='settled' AND source->>'kind'<>'continuation' ORDER BY ordinal`
