package pgstore

// dependencyResultSourceSQL selects a precise accepted attempt in the finding
// receipt's original execution. Subject matching cannot choose between batch
// members: ambiguity is returned to the caller. The primary/FK indexes bound
// all reads to the one execution and its at most 64 members; unrelated Audit
// items and Project bindings are never examined.
const dependencyResultSourceSQL = `
SELECT execution.execution_id, execution.round_id, member.item_id, member.item_attempt,
       COALESCE(execution.run_id, execution.run_provenance->>'runId'),
       execution.collection_source_output_ref, execution.collection_source_output_digest,
       link.artifact_ref, link.artifact_digest, link.media_type, link.size_bytes,
       CASE WHEN jsonb_array_length(execution.collection_retained_refs)<=32 AND octet_length(execution.collection_retained_refs::text)<=1048576 THEN execution.collection_retained_refs ELSE NULL END,
       jsonb_array_length(execution.collection_retained_refs)>32 OR octet_length(execution.collection_retained_refs::text)>1048576
FROM audit_executions AS execution
JOIN audit_execution_items AS member USING (execution_id)
JOIN audit_items AS item ON item.item_id = member.item_id AND item.audit_id = execution.audit_id
JOIN audit_artifact_links AS link
  ON link.audit_id = execution.audit_id AND link.logical_key = 'result/' || member.execution_item_id
WHERE execution.audit_id = $1 AND execution.execution_id = $2
  AND execution.role = 'check' AND execution.collection_disposition = 'accepted-result'
  AND member.collection_disposition = 'accepted-result'
  AND item.accepted_result_ref = member.result_ref
  AND item.accepted_result_digest = member.result_digest
  AND (item.subject_key = $3 OR (SELECT count(*) FROM audit_execution_items WHERE execution_id = $2) = 1)
ORDER BY member.batch_ordinal LIMIT 2`

// retainedDependencyCostSQL is the indexed distinct-revision debit used by
// both atomic admission and definite budget refusal classification.
const retainedDependencyCostSQL = `
    SELECT COALESCE(sum((artifact->>'sizeBytes')::bigint),0)::bigint AS bytes
    FROM (SELECT DISTINCT artifact FROM retained_input) AS planned
    WHERE NOT EXISTS (SELECT 1 FROM audit_artifact_links AS existing
        WHERE existing.audit_id=$1
          AND existing.artifact_ref->>'namespace'=planned.artifact #>> '{ref,namespace}'
          AND existing.artifact_ref->>'name'=planned.artifact #>> '{ref,name}'
          AND existing.artifact_ref->>'revision'=planned.artifact #>> '{ref,revision}'
          AND existing.artifact_digest=planned.artifact->>'digest')`

// dependencyBudgetExhaustedSQL classifies the budget refusal after an atomic
// admission stored nothing; it uses the same cost definition and exact index.
const dependencyBudgetExhaustedSQL = `
WITH retained_input AS MATERIALIZED (
 SELECT * FROM jsonb_to_recordset($2::jsonb) AS link(artifact jsonb)
), retained_cost AS MATERIALIZED (` + retainedDependencyCostSQL + `)
SELECT EXISTS(SELECT 1 FROM audits AS audit CROSS JOIN retained_cost
 WHERE audit.audit_id=$1 AND audit.state='active' AND audit.dispatch_state='open'
 AND retained_cost.bytes>audit.max_evidence_bytes-audit.retained_evidence_bytes)`
