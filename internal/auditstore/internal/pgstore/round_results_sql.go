package pgstore

// roundResultsItemsSQL reads only the next selected Round's indexed primary
// rows. Its joins select each item's last authoritative attempt/receipt and
// the accepted result's already retained descriptor; no live Run is needed.
const roundResultsItemsSQL = `
WITH candidates AS MATERIALIZED (
    SELECT item_id, ordinal, results_projection_bytes FROM audit_items
     WHERE audit_id=$1 AND round_id=$2 AND ordinal>$3 ORDER BY ordinal LIMIT 32
), selected AS (
    SELECT item_id, sum(results_projection_bytes) OVER (ORDER BY ordinal) AS bytes FROM candidates
)
SELECT item.item_id, item.item_key, item.ordinal, item.kind, item.subject_key,
       item.state, item.final_disposition, item.coverage_status,
       item.coverage_requested, item.coverage_completed, item.coverage_gaps, item.coverage_rationale,
       item.accepted_result_ref, item.accepted_result_digest, result.media_type, result.size_bytes,
       member.execution_item_id, member.execution_id, member.item_attempt,
       execution.collection_receipt_id, member.collection_disposition, execution.terminal_outcome,
       execution.run_id, execution.collection_source_output_ref, execution.collection_source_output_digest
  FROM audit_items AS item
  JOIN selected ON selected.item_id=item.item_id AND selected.bytes<=524288
  LEFT JOIN audit_execution_items AS member ON member.execution_item_id = item.last_execution_item_id
  LEFT JOIN audit_executions AS execution ON execution.execution_id = member.execution_id
  LEFT JOIN audit_artifact_links AS result ON result.audit_id = item.audit_id
       AND result.logical_key = 'result/' || member.execution_item_id
 WHERE item.audit_id = $1 AND item.round_id = $2 AND item.ordinal > $3
 ORDER BY item.ordinal
 LIMIT 32`

// roundResultsAttachmentsSQL seeks one selected attempt's already accepted
// evidence descriptors using the attribution index. LIMIT is the remaining
// descriptor budget; the caller leaves the final item for the next slice if
// a limit may have cut its closure. No containing-Audit JSON scan is used.
const roundResultsAttachmentsSQL = `
WITH candidates AS MATERIALIZED (
    SELECT logical_key, results_descriptor_bytes FROM audit_artifact_links
     WHERE audit_id=$1 AND COALESCE(source_provenance->>'executionItemId',source_provenance->'source'->>'executionItemId')=$2
       AND logical_key LIKE 'evidence/%' ORDER BY logical_key LIMIT $3
), selected AS (
    SELECT logical_key, sum(results_descriptor_bytes) OVER(ORDER BY logical_key) AS bytes FROM candidates
)
SELECT selected.bytes<=$4, CASE WHEN selected.bytes<=$4 THEN link.logical_key END,
       CASE WHEN selected.bytes<=$4 THEN link.artifact_ref END,
       CASE WHEN selected.bytes<=$4 THEN link.artifact_digest END,
       CASE WHEN selected.bytes<=$4 THEN link.media_type END,
       CASE WHEN selected.bytes<=$4 THEN link.size_bytes END, link.results_descriptor_bytes,
       CASE WHEN selected.bytes<=$4 THEN COALESCE(link.source_provenance->'evidence'->>'kind','') END,
       CASE WHEN selected.bytes<=$4 THEN COALESCE(link.source_provenance->'evidence'->>'id','') END,
       CASE WHEN selected.bytes<=$4 THEN COALESCE(link.source_provenance->'evidence'->>'content_member_id','') END,
       CASE WHEN selected.bytes<=$4 THEN link.display_ref END,
       CASE WHEN selected.bytes<=$4 THEN link.source_provenance->'evidence'->'artifact' END
  FROM selected JOIN audit_artifact_links AS link ON link.audit_id=$1 AND link.logical_key=selected.logical_key
 ORDER BY selected.logical_key`

// roundResultsFindingsSQL seeks only assessments of the selected attempt.
// The persisted current machine assessment and human decision are copied as
// revision pins, with no inference from the Run's success.
const roundResultsFindingsSQL = `SELECT DISTINCT finding.finding_id,finding.project_id
 FROM audit_execution_items AS member
 CROSS JOIN LATERAL jsonb_to_recordset(member.finding_sources) AS association(receipt_id text)
 JOIN project_finding_sources AS source USING(receipt_id)
 JOIN project_findings AS finding USING(project_id,finding_id)
 WHERE member.audit_id=$1 AND member.execution_item_id=$2
 ORDER BY finding.finding_id LIMIT $3`

// publishRoundResultsLinkSQL inserts an immutable retained metadata receipt
// under its natural logical key. The Audit lock held by the caller serializes
// the distinct revision evidence debit; replays charge no bytes.
const publishRoundResultsLinkSQL = `
INSERT INTO audit_artifact_links
    (audit_id, logical_key, artifact_ref, artifact_digest, media_type, size_bytes, source_provenance)
VALUES ($1, $2, $3::jsonb, $4, $5, $6,
    jsonb_build_object('schema', 'contractor.audit.round-results-provenance.v1',
        'roundId', $7::text, 'sourceGeneration', $8::bigint))
ON CONFLICT (audit_id, logical_key) DO NOTHING`
