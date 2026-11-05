package pgstore

// collectionEvents record the collected execution $4 with disposition $6.
var collectionEvents = auditEvents(auditEvent("'execution.collected'", "$4::text", "NULL",
	"jsonb_build_object('disposition', $6::text)"))

// collectAuditExecutionSQL is one atomic statement. live_claim and execution_gate
// lock the authority rows; member/finding validation gates advanced_audit and its
// evidence budget. Every receipt, settlement, coverage, association, link and
// event mutation depends on that gate. Keep these CTE dependencies and lock order
// together; an incomplete batch must leave no partial durable effects.
var collectAuditExecutionSQL = `
WITH collection_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($10::jsonb) AS item(
        execution_item_id text, disposition text, result_ref jsonb,
        result_digest text, retryable boolean, task_id text, task_result jsonb, gap_code text
    )
), ` + collectionLinkStateCTEs("$11") + `, live_claim AS MATERIALIZED (
    ` + auditClaims.LiveClaimSQL("$1", "$2", "$3") + `
), execution_gate AS MATERIALIZED (
    SELECT execution.*, audit.max_item_run_attempts
      FROM audit_executions AS execution
      JOIN audits AS audit USING (audit_id)
      JOIN live_claim USING (audit_id)
     WHERE execution.audit_id = $1 AND execution.execution_id = $4
       AND execution.state = 'collecting'
       AND (
           (execution.terminal_outcome = 'succeeded' AND $6 IN ('accepted-result', 'missing-output', 'invalid-result'))
           OR (execution.terminal_outcome IN ('failed', 'submission-failed') AND $6 = 'execution-failed')
           OR (execution.terminal_outcome = 'cancelled' AND $6 = 'execution-cancelled')
           OR $6 = 'collection-contract-invalid'
       )
     FOR UPDATE OF audit, execution
), member_validation AS MATERIALIZED (
    SELECT count(member.execution_item_id)::integer AS stored_count,
           count(input.execution_item_id) FILTER (
               WHERE task.task_id IS NOT NULL
           )::integer AS matched_count
      FROM audit_execution_items AS member
      JOIN execution_gate AS execution USING (execution_id)
      LEFT JOIN collection_input AS input
        ON input.execution_item_id = member.execution_item_id
       AND input.disposition = $6
       AND member.state = 'collecting'
       AND member.task_id = input.task_id
      LEFT JOIN audit_tasks AS task
        ON task.task_id=member.task_id AND task.audit_id=member.audit_id
       AND task.last_execution_id=member.execution_id AND task.attempts=member.item_attempt
       AND task.state='selected'
), advanced_audit AS (
    UPDATE audits AS audit
       SET retained_evidence_bytes = audit.retained_evidence_bytes + link_cost.retained_bytes,
           ` + collectionEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM execution_gate, member_validation, link_cost
     WHERE audit.audit_id = execution_gate.audit_id
       AND member_validation.stored_count = jsonb_array_length($10::jsonb)
       AND member_validation.matched_count = member_validation.stored_count
       AND link_cost.valid AND link_cost.retained_bytes <= $12::bigint
       AND audit.retained_evidence_bytes + link_cost.retained_bytes <= audit.max_evidence_bytes
    RETURNING audit.audit_id, audit.max_item_run_attempts, ` + collectionEvents.appended() + `
), inserted_receipt AS (
    -- Collecting records the receipt on the execution and moves it to
    -- 'collected' in one row update.
    UPDATE audit_executions AS execution
       SET state = 'collected', updated_at = clock_timestamp(),
           collection_receipt_id = $5, collection_disposition = $6,
           collection_source_output_ref = $7::jsonb, collection_source_output_digest = $8,
           collection_retained_refs = $9::jsonb, collection_error_code = $13,
           collection_request_digest = $14, collected_at = clock_timestamp()
      FROM execution_gate AS gate JOIN advanced_audit USING (audit_id)
     WHERE execution.execution_id = gate.execution_id AND execution.state = 'collecting'
    RETURNING ` + collectionReceiptProjection("execution") + `
), settled_attempts AS (
    UPDATE audit_execution_items AS member
       SET state = 'settled', collection_disposition = input.disposition,
           result_ref = input.result_ref, result_digest = input.result_digest,
           collected_at = clock_timestamp()
      FROM collection_input AS input, inserted_receipt AS receipt
     WHERE member.execution_id = receipt.execution_id
       AND member.execution_item_id = input.execution_item_id
       AND member.state = 'collecting'
    RETURNING member.execution_item_id, member.item_attempt, member.task_id
), task_settlement AS MATERIALIZED (
    SELECT attempt.task_id, receipt.audit_id, receipt.execution_id,
           input.task_result,input.gap_code,
           CASE WHEN input.retryable AND attempt.item_attempt<LEAST(advanced.max_item_run_attempts,
                    COALESCE((audit.profile_snapshot #>> ARRAY['program','workflows',selected.workflow_alias,'max_attempts'])::integer,advanced.max_item_run_attempts))
                  AND receipt.disposition='execution-failed'
                  AND audit.state IN ('active','paused') THEN 'ready' ELSE 'settled' END AS next_state
      FROM settled_attempts AS attempt
      JOIN collection_input AS input USING(execution_item_id)
      CROSS JOIN advanced_audit AS advanced
      CROSS JOIN inserted_receipt AS receipt
      JOIN audits AS audit ON audit.audit_id=receipt.audit_id
      JOIN audit_tasks AS selected ON selected.task_id=attempt.task_id AND selected.audit_id=receipt.audit_id
), settled_tasks AS (
    UPDATE audit_tasks AS task
       SET state=input.next_state,
           result=CASE WHEN input.next_state='ready' THEN NULL ELSE input.task_result::json END,
           gap_code=CASE WHEN input.next_state='ready' THEN NULL ELSE input.gap_code END,
           updated_at=clock_timestamp()
      FROM task_settlement AS input
     WHERE task.task_id=input.task_id AND task.audit_id=input.audit_id
       AND task.state='selected' AND task.last_execution_id=input.execution_id
), inserted_links AS (
    INSERT INTO audit_artifact_links (
        audit_id, artifact_scope, logical_key, artifact_ref, artifact_digest,
        media_type, size_bytes, source_provenance, display_ref
    )
    SELECT receipt.audit_id, link.artifact_scope, link.logical_key, link.artifact_ref,
           link.artifact_digest, link.media_type, link.size_bytes,
           link.source_provenance, link.display_ref
      FROM inserted_receipt AS receipt CROSS JOIN link_input AS link
    ON CONFLICT (audit_id,logical_key) DO NOTHING
)
SELECT ` + prefixedReceiptColumns("inserted_receipt") + `
  FROM inserted_receipt`

// collectionLinkStateCTEs validates repeated logical links by their complete
// immutable descriptor/provenance, and charges each retained revision once.
// Both lookups seek existing Audit indexes; each request has bounded links.
func collectionLinkStateCTEs(input string) string {
	return `link_input AS MATERIALIZED (
 SELECT * FROM jsonb_to_recordset(` + input + `::jsonb) AS link(
  artifact_scope text,logical_key text,artifact_ref jsonb,artifact_digest text,
  media_type text,size_bytes bigint,source_provenance jsonb,display_ref text)
), link_validation AS MATERIALIZED (
 SELECT COALESCE(bool_and(existing.logical_key IS NULL OR
  ROW(existing.artifact_scope,existing.artifact_ref,existing.artifact_digest,
      existing.media_type,existing.size_bytes,existing.source_provenance,existing.display_ref)
  IS NOT DISTINCT FROM ROW(input.artifact_scope,input.artifact_ref,input.artifact_digest,
      input.media_type,input.size_bytes,input.source_provenance,input.display_ref)),true) AS valid
 FROM link_input AS input LEFT JOIN LATERAL (
  SELECT * FROM audit_artifact_links AS stored
  WHERE stored.audit_id=$1 AND stored.logical_key=input.logical_key LIMIT 1
 ) AS existing ON true
), distinct_material AS MATERIALIZED (
 SELECT DISTINCT ON (artifact_scope,artifact_ref) artifact_scope,artifact_ref,artifact_digest,media_type,size_bytes
 FROM link_input ORDER BY artifact_scope,artifact_ref,logical_key
), link_cost AS MATERIALIZED (
 SELECT (SELECT valid FROM link_validation) AND COALESCE(bool_and(
   existing.artifact_digest IS NULL OR
   ROW(existing.artifact_digest,existing.media_type,existing.size_bytes)
    IS NOT DISTINCT FROM ROW(input.artifact_digest,input.media_type,input.size_bytes)),true) AS valid,
  COALESCE(sum(input.size_bytes) FILTER(WHERE existing.artifact_digest IS NULL),0)::bigint AS retained_bytes
 FROM distinct_material AS input LEFT JOIN LATERAL (
  (SELECT point.artifact_digest,point.media_type,point.size_bytes FROM (
   SELECT stored.artifact_ref,stored.artifact_digest,stored.media_type,stored.size_bytes
   FROM audit_artifact_links AS stored WHERE input.artifact_scope='audit'
    AND stored.artifact_scope='audit' AND stored.audit_id=$1
    AND (stored.artifact_ref->>'namespace',stored.artifact_ref->>'name',stored.artifact_ref->>'revision')
       >= (input.artifact_ref->>'namespace',input.artifact_ref->>'name',input.artifact_ref->>'revision')
   ORDER BY stored.artifact_ref->>'namespace',stored.artifact_ref->>'name',
            stored.artifact_ref->>'revision',stored.logical_key LIMIT 1
  ) AS point WHERE point.artifact_ref=input.artifact_ref)
  UNION ALL
  (SELECT stored.artifact_digest,stored.media_type,stored.size_bytes
   FROM audit_artifact_links AS stored WHERE input.artifact_scope='project'
    AND stored.artifact_scope='project' AND stored.audit_id=$1
    AND stored.artifact_ref->>'namespace'=input.artifact_ref->>'namespace'
    AND stored.artifact_ref->>'name'=input.artifact_ref->>'name'
    AND stored.artifact_ref->>'revision'=input.artifact_ref->>'revision'
   ORDER BY stored.artifact_digest LIMIT 1)
 ) AS existing ON true
)`
}

var readCollectionLinkStateSQL = `WITH ` + collectionLinkStateCTEs("$2") + `
SELECT valid,retained_bytes FROM link_cost`
