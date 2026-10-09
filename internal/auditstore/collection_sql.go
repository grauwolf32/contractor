package auditstore

// collectAuditExecutionSQL is one atomic statement. live_claim and execution_gate
// lock the authority rows; member/finding validation gates advanced_audit and its
// evidence budget. Every receipt, settlement, coverage, association, link and
// event mutation depends on that gate. Keep these CTE dependencies and lock order
// together; an incomplete batch must leave no partial durable effects.
// stale_finding_reviews locks review requests only after execution_gate holds
// the Audit row, the lock every review mutation takes first.
var collectAuditExecutionSQL = `
WITH collection_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($10::jsonb) AS item(
        execution_item_id text, disposition text, result_ref jsonb,
        result_digest text, retryable boolean, final_disposition text,
        status text, requested jsonb, completed jsonb, gaps jsonb, rationale text,
        finding_associations jsonb
    )
), live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
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
           count(input.execution_item_id)::integer AS matched_count
      FROM audit_execution_items AS member
      JOIN execution_gate AS execution USING (execution_id)
      LEFT JOIN collection_input AS input
        ON input.execution_item_id = member.execution_item_id
       AND input.disposition = $6
       AND member.state = 'collecting'
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
      LEFT JOIN audit_finding_contributions AS contribution
       ON contribution.audit_id = $1 AND contribution.receipt_id = input.receipt_id
       AND contribution.proposal_ref #> '{ref}' = input.proposal_ref
       AND contribution.proposal_ref #>> '{digest}' = input.proposal_digest
       AND contribution.proposal_ref #>> '{mediaType}' = input.proposal_media_type
       AND (contribution.proposal_ref #>> '{sizeBytes}')::bigint = input.proposal_size_bytes
      LEFT JOIN audit_execution_items AS member
       ON member.execution_item_id = input.execution_item_id
       AND member.execution_id = $4 AND member.audit_id = $1
), stale_finding_reviews AS MATERIALIZED (
    -- Each associated finding gets a new assessment and revision below, so a
    -- pending triage request for its earlier revision can never be decided.
    -- The locked rows fix how many review.expired events this statement adds.
    SELECT request.request_id, request.finding_id
      FROM audit_review_requests AS request
      JOIN execution_gate AS gate ON gate.audit_id = request.audit_id
     WHERE request.subject_kind = 'finding' AND request.kind = 'finding-triage'
       AND request.state = 'pending'
       AND request.finding_id IN (
           SELECT contribution.finding_id
             FROM finding_input AS input
             JOIN audit_finding_contributions AS contribution
               ON contribution.audit_id = $1 AND contribution.receipt_id = input.receipt_id
       )
     FOR UPDATE OF request
), stale_review_count AS MATERIALIZED (
    SELECT count(*)::bigint AS expired FROM stale_finding_reviews
), advanced_audit AS (
    UPDATE audits AS audit
       SET retained_evidence_bytes = audit.retained_evidence_bytes + $12,
           revision = audit.revision + 1 + stale.expired,
           next_event_sequence = audit.next_event_sequence + 1 + stale.expired,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM execution_gate, member_validation, finding_validation,
           stale_review_count AS stale
     WHERE audit.audit_id = execution_gate.audit_id
       AND member_validation.stored_count = jsonb_array_length($10::jsonb)
       AND member_validation.matched_count = member_validation.stored_count
       AND finding_validation.input_count = finding_validation.matched_count
       AND audit.retained_evidence_bytes + $12 <= audit.max_evidence_bytes
    RETURNING audit.audit_id, audit.max_item_run_attempts,
              audit.next_event_sequence - 1 - stale.expired AS collected_sequence
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
           collected_at = clock_timestamp()
      FROM collection_input AS input, inserted_receipt AS receipt
     WHERE member.execution_id = receipt.execution_id
       AND member.execution_item_id = input.execution_item_id
       AND member.state = 'collecting'
    RETURNING member.execution_item_id, member.item_id, member.item_attempt
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
), inserted_finding_assessments AS (
    INSERT INTO audit_finding_assessments (
        assessment_id, finding_id, audit_id, receipt_id, item_id,
        execution_item_id, collection_receipt_id, semantic_assessment,
        result_ref, result_digest
    )
    SELECT input.assessment_id, contribution.finding_id, receipt.audit_id,
           input.receipt_id, member.item_id, input.execution_item_id,
           receipt.receipt_id, input.semantic_assessment,
           collected.result_ref, collected.result_digest
      FROM finding_input AS input
      JOIN audit_finding_contributions AS contribution
       ON contribution.audit_id = $1 AND contribution.receipt_id = input.receipt_id
       AND contribution.proposal_ref #> '{ref}' = input.proposal_ref
       AND contribution.proposal_ref #>> '{digest}' = input.proposal_digest
       AND contribution.proposal_ref #>> '{mediaType}' = input.proposal_media_type
       AND (contribution.proposal_ref #>> '{sizeBytes}')::bigint = input.proposal_size_bytes
      JOIN audit_execution_items AS member
        ON member.execution_item_id = input.execution_item_id
       AND member.execution_id = $4
      JOIN collection_input AS collected
        ON collected.execution_item_id = input.execution_item_id
      CROSS JOIN inserted_receipt AS receipt
    RETURNING assessment_id, finding_id, audit_id, execution_item_id
), current_finding_assessments AS (
    -- Batch members may verify distinct checks of one finding. Each keeps its
    -- assessment; the highest batch ordinal becomes current, as if the members
    -- had been collected in batch order, so the finding changes once.
    SELECT DISTINCT ON (assessment.finding_id)
           assessment.assessment_id, assessment.finding_id, assessment.audit_id
      FROM inserted_finding_assessments AS assessment
      JOIN audit_execution_items AS member
        ON member.execution_item_id = assessment.execution_item_id
     ORDER BY assessment.finding_id, member.batch_ordinal DESC
), updated_findings AS (
    UPDATE audit_findings AS finding
       SET current_assessment_id = assessment.assessment_id,
           current_decision_id = NULL, state = 'proposed',
           rejection_reason = NULL, duplicate_target_id = NULL,
           revision = finding.revision + 1,
           updated_at = GREATEST(clock_timestamp(), finding.updated_at + interval '1 microsecond')
      FROM current_finding_assessments AS assessment
     WHERE finding.finding_id = assessment.finding_id
       AND finding.audit_id = assessment.audit_id
), link_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($11::jsonb) AS link(
        logical_key text, artifact_ref jsonb, artifact_digest text,
        media_type text, size_bytes bigint, source_provenance jsonb, display_ref text
    )
), inserted_links AS (
    INSERT INTO audit_artifact_links (
        audit_id, logical_key, artifact_ref, artifact_digest,
        media_type, size_bytes, source_provenance, display_ref
    )
    SELECT receipt.audit_id, link.logical_key, link.artifact_ref,
           link.artifact_digest, link.media_type, link.size_bytes,
           link.source_provenance, link.display_ref
      FROM inserted_receipt AS receipt CROSS JOIN link_input AS link
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, summary)
    SELECT receipt.audit_id, advanced.collected_sequence,
           'execution.collected', receipt.execution_id,
           jsonb_build_object('disposition', receipt.disposition)
      FROM inserted_receipt AS receipt JOIN advanced_audit AS advanced USING (audit_id)
), expired_finding_reviews AS (
    UPDATE audit_review_requests AS request
       SET state = 'expired', revision = request.revision + 1,
           updated_at = GREATEST(clock_timestamp(), request.updated_at + interval '1 microsecond')
      FROM stale_finding_reviews AS stale, advanced_audit
     WHERE request.request_id = stale.request_id AND request.state = 'pending'
), expired_review_events AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, summary)
    SELECT advanced.audit_id,
           advanced.collected_sequence + row_number() OVER (ORDER BY stale.request_id),
           'review.expired', stale.request_id,
           jsonb_build_object(
               'subjectKind', 'finding', 'findingId', stale.finding_id,
               'kind', 'finding-triage'
           )
      FROM stale_finding_reviews AS stale CROSS JOIN advanced_audit AS advanced
)
SELECT ` + prefixedReceiptColumns("inserted_receipt") + `
  FROM inserted_receipt`
