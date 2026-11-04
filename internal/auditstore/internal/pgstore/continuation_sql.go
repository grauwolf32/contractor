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
   ORDER BY ordinal LIMIT 16
 ) candidate
), selected AS MATERIALIZED (
 SELECT * FROM candidates ORDER BY ordinal LIMIT 16
)
SELECT task_id,ordinal,hypothesis_snapshot,result,gap_code
 FROM audit_tasks JOIN selected USING(task_id,ordinal)
 WHERE state='settled' ORDER BY ordinal`
