package pgstore

// listResearchMaterialsSQL seeks accepted links before budgeting metadata.
// Provenance payloads are deliberately not selected or decoded by this list.
const listResearchMaterialsSQL = `
WITH candidates AS MATERIALIZED (
 SELECT material_ordinal,logical_key,artifact_scope,artifact_ref,artifact_digest,media_type,size_bytes,created_at,
        octet_length(artifact_ref::text)+octet_length(to_jsonb(logical_key)::text)+octet_length(to_jsonb(media_type)::text)+2048 AS bytes
 FROM audit_artifact_links WHERE audit_id=$1 AND material_ordinal>$2
 ORDER BY material_ordinal LIMIT $3
), bounded AS (
 SELECT candidates.*,sum(bytes) OVER (ORDER BY material_ordinal) AS used_bytes,
        count(*) OVER () AS candidate_count FROM candidates
)
SELECT material_ordinal,logical_key,artifact_scope,artifact_ref,artifact_digest,media_type,size_bytes,created_at,candidate_count
 FROM bounded WHERE used_bytes<=1044480 ORDER BY material_ordinal`

// listTaskAttemptsSQL seeks the bounded ordinary Task membership history;
// the execution's retained Run ID survives deletion of the source Run.
const listTaskAttemptsSQL = `
SELECT member.item_attempt,member.execution_id,execution.run_id,
       execution.run_deleted_at IS NOT NULL,execution.terminal_outcome,
       member.collection_disposition,member.state,
       member.created_at,member.collected_at
 FROM audit_execution_items AS member
 JOIN audit_executions AS execution USING(audit_id,execution_id)
 WHERE member.audit_id=$1 AND member.task_id=$2 AND member.item_attempt>$3
 ORDER BY member.item_attempt LIMIT $4`
