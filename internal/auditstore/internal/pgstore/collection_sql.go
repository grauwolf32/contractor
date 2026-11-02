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
        result_digest text, retryable boolean, final_disposition text,
        status text, requested jsonb, completed jsonb, gaps jsonb, rationale text,
        finding_associations jsonb, task_id text, task_result jsonb, gap_code text
    )
), live_claim AS MATERIALIZED (
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
               WHERE member.task_id IS NULL OR task.task_id IS NOT NULL
           )::integer AS matched_count
      FROM audit_execution_items AS member
      JOIN execution_gate AS execution USING (execution_id)
      LEFT JOIN collection_input AS input
        ON input.execution_item_id = member.execution_item_id
       AND input.disposition = $6
       AND member.state = 'collecting'
       AND member.task_id IS NOT DISTINCT FROM input.task_id
      LEFT JOIN audit_tasks AS task
        ON task.task_id=member.task_id AND task.audit_id=member.audit_id
       AND task.last_execution_id=member.execution_id AND task.attempts=member.item_attempt
       AND task.state='selected'
), finding_input AS MATERIALIZED (
    SELECT item.execution_item_id, association.*
      FROM collection_input AS item
      CROSS JOIN LATERAL jsonb_to_recordset(item.finding_associations) AS association(
          assessment_id text, receipt_id text, proposal_ref jsonb,
          proposal_digest text, proposal_media_type text,
          proposal_size_bytes bigint, semantic_assessment text
      )
), finding_validation AS MATERIALIZED (
    SELECT count(input.receipt_id)::integer AS input_count,
           count(*) FILTER (
               WHERE contribution.receipt_id IS NOT NULL
                 AND member.execution_item_id IS NOT NULL
           )::integer AS matched_count
      FROM finding_input AS input
      LEFT JOIN finding_proposal_audit_holds AS contribution
       ON contribution.audit_id = $1 AND contribution.receipt_id = input.receipt_id
       AND contribution.proposal_ref #> '{ref}' = input.proposal_ref
       AND contribution.proposal_ref #>> '{digest}' = input.proposal_digest
       AND contribution.proposal_ref #>> '{mediaType}' = input.proposal_media_type
       AND (contribution.proposal_ref #>> '{sizeBytes}')::bigint = input.proposal_size_bytes
      LEFT JOIN audit_execution_items AS member
       ON member.execution_item_id = input.execution_item_id
       AND member.execution_id = $4 AND member.audit_id = $1
), advanced_audit AS (
    UPDATE audits AS audit
       SET retained_evidence_bytes = audit.retained_evidence_bytes + $12,
           ` + collectionEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM execution_gate, member_validation, finding_validation
     WHERE audit.audit_id = execution_gate.audit_id
       AND member_validation.stored_count = jsonb_array_length($10::jsonb)
       AND member_validation.matched_count = member_validation.stored_count
       AND finding_validation.input_count = finding_validation.matched_count
       AND audit.retained_evidence_bytes + $12 <= audit.max_evidence_bytes
    RETURNING audit.audit_id, audit.max_item_run_attempts, ` + collectionEvents.appended() + `
), inserted_receipt AS (
    -- Collecting records the receipt on the execution and moves it to
    -- 'collected' in one row update.
    UPDATE audit_executions AS execution
       SET state = 'collected', updated_at = clock_timestamp(),
           collection_receipt_id = $5, collection_disposition = $6,
           collection_source_output_ref = $7::jsonb, collection_source_output_digest = $8,
           collection_retained_refs = $9::jsonb, collection_error_code = $13,
           collection_request_digest = $14, collected_at = clock_timestamp(),
           preparation_outputs = $15::jsonb
      FROM execution_gate AS gate JOIN advanced_audit USING (audit_id)
     WHERE execution.execution_id = gate.execution_id AND execution.state = 'collecting'
    RETURNING ` + collectionReceiptProjection("execution") + `
), settled_attempts AS (
    UPDATE audit_execution_items AS member
       SET state = 'settled', collection_disposition = input.disposition,
           result_ref = input.result_ref, result_digest = input.result_digest,
           finding_sources=input.finding_associations,
           collected_at = clock_timestamp()
      FROM collection_input AS input, inserted_receipt AS receipt
     WHERE member.execution_id = receipt.execution_id
       AND member.execution_item_id = input.execution_item_id
       AND member.state = 'collecting'
    RETURNING member.execution_item_id, member.item_id, member.item_attempt, member.task_id
), task_settlement AS MATERIALIZED (
    SELECT attempt.task_id, receipt.audit_id, receipt.execution_id,
           input.task_result,input.gap_code,
           CASE WHEN input.retryable AND attempt.item_attempt<advanced.max_item_run_attempts
                  AND receipt.disposition='execution-failed'
                  AND audit.state IN ('active','paused') THEN 'ready' ELSE 'settled' END AS next_state
      FROM settled_attempts AS attempt
      JOIN collection_input AS input USING(execution_item_id)
      CROSS JOIN advanced_audit AS advanced
      CROSS JOIN inserted_receipt AS receipt
      JOIN audits AS audit ON audit.audit_id=receipt.audit_id
     WHERE attempt.task_id IS NOT NULL
), settled_tasks AS (
    UPDATE audit_tasks AS task
       SET state=input.next_state,
           result=CASE WHEN input.next_state='ready' THEN NULL ELSE input.task_result::json END,
           gap_code=CASE WHEN input.next_state='ready' THEN NULL ELSE input.gap_code END,
           updated_at=clock_timestamp()
      FROM task_settlement AS input
     WHERE task.task_id=input.task_id AND task.audit_id=input.audit_id
       AND task.state='selected' AND task.last_execution_id=input.execution_id
), settled_items AS (
    -- One update per item row: the item settles only while it is still
    -- collecting, and its coverage always records the collected attempt.
    UPDATE audit_items AS item
       SET state = CASE
               WHEN item.state <> 'collecting' THEN item.state
               WHEN input.retryable
                AND attempt.item_attempt < advanced.max_item_run_attempts
                AND input.disposition NOT IN ('accepted-result', 'execution-cancelled')
                 THEN 'ready'
               ELSE 'settled'
           END,
           final_disposition = CASE
               WHEN item.state <> 'collecting' THEN item.final_disposition
               WHEN input.retryable
                AND attempt.item_attempt < advanced.max_item_run_attempts
                AND input.disposition NOT IN ('accepted-result', 'execution-cancelled')
                 THEN NULL
               ELSE input.final_disposition
           END,
           accepted_result_ref = CASE
               WHEN item.state <> 'collecting' THEN item.accepted_result_ref
               WHEN input.disposition = 'accepted-result' THEN input.result_ref ELSE NULL END,
           accepted_result_digest = CASE
               WHEN item.state <> 'collecting' THEN item.accepted_result_digest
               WHEN input.disposition = 'accepted-result' THEN input.result_digest ELSE NULL END,
           updated_at = CASE
               WHEN item.state <> 'collecting' THEN item.updated_at
               ELSE GREATEST(clock_timestamp(), item.updated_at + interval '1 microsecond') END,
           coverage_status = input.status,
           coverage_requested = CASE WHEN input.disposition = 'collection-contract-invalid'
               THEN item.coverage_requested ELSE input.requested END,
           coverage_completed = CASE WHEN input.disposition = 'collection-contract-invalid'
               THEN item.coverage_completed ELSE input.completed END,
           coverage_gaps = CASE WHEN input.disposition = 'collection-contract-invalid'
               THEN CASE WHEN item.coverage_gaps ? 'collection-contract-invalid'
                   THEN item.coverage_gaps
                   ELSE item.coverage_gaps || jsonb_build_array('collection-contract-invalid') END
               ELSE input.gaps END,
           coverage_rationale = input.rationale,
           coverage_result_ref = input.result_ref, coverage_result_digest = input.result_digest,
           coverage_updated_at = clock_timestamp()
      FROM settled_attempts AS attempt
      JOIN collection_input AS input USING (execution_item_id)
      CROSS JOIN advanced_audit AS advanced
     WHERE item.item_id = attempt.item_id
), link_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($11::jsonb) AS link(
        artifact_scope text, logical_key text, artifact_ref jsonb, artifact_digest text,
        media_type text, size_bytes bigint, source_provenance jsonb, display_ref text
    )
), inserted_links AS (
    INSERT INTO audit_artifact_links (
        audit_id, artifact_scope, logical_key, artifact_ref, artifact_digest,
        media_type, size_bytes, source_provenance, display_ref
    )
    SELECT receipt.audit_id, link.artifact_scope, link.logical_key, link.artifact_ref,
           link.artifact_digest, link.media_type, link.size_bytes,
           link.source_provenance, link.display_ref
      FROM inserted_receipt AS receipt CROSS JOIN link_input AS link
)
SELECT ` + prefixedReceiptColumns("inserted_receipt") + `
  FROM inserted_receipt`
