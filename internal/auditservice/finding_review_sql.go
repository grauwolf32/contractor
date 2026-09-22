package auditservice

// Finding review statements. See finding_review.go for the callers.

// listFindingProvenanceSQL walks one finding back to every record that
// produced it: assessments anchor the receipts and execution items, and
// the union is paged by (created_at, record_id) so a caller resuming from
// a cursor sees each record exactly once.
var listFindingProvenanceSQL = `
WITH anchors AS (
    SELECT DISTINCT assessment.finding_id, assessment.audit_id,
           assessment.receipt_id, assessment.item_id
      FROM audit_finding_assessments AS assessment
     WHERE assessment.audit_id = $1 AND assessment.finding_id = $2
       AND assessment.item_id IS NOT NULL
), records AS (
    SELECT contribution.created_at, 'proposal:' || contribution.receipt_id AS record_id,
           'source-proposal'::text AS kind, contribution.receipt_id,
           contribution.relation, contribution.proposal_ref,
           NULL::text AS assessment_id, NULL::text AS semantic_assessment,
           NULL::jsonb AS result_ref, NULL::text AS result_digest,
           NULL::text AS item_id, NULL::text AS execution_item_id,
           NULL::text AS collection_receipt_id, false AS direct_verification,
           NULL::jsonb AS contract_ref, NULL::text AS contract_digest,
           NULL::text AS execution_id, NULL::text AS execution_role,
           NULL::integer AS item_attempt, NULL::text AS item_state,
           NULL::text AS collection_disposition, NULL::text AS terminal_outcome,
           NULL::text AS run_id, NULL::jsonb AS run_provenance,
           false AS run_deleted, NULL::jsonb AS task_ref,
           NULL::text AS task_digest, NULL::jsonb AS item_origin,
           NULL::text AS workflow_role, NULL::timestamptz AS collected_at,
           NULL::timestamptz AS assessment_accepted_at
      FROM audit_finding_contributions AS contribution
     WHERE contribution.audit_id = $1 AND contribution.finding_id = $2
    UNION ALL
    SELECT member.created_at,
           'attempt:' || anchor.receipt_id || ':' || member.execution_item_id,
           'check-attempt'::text,
           anchor.receipt_id, 'verification'::text, contribution.proposal_ref,
           selected.assessment_id, selected.semantic_assessment,
           member.result_ref, member.result_digest,
           member.item_id, member.execution_item_id,
           selected.collection_receipt_id, false,
           NULL::jsonb, NULL::text,
           execution.execution_id, execution.role, member.item_attempt,
           member.state, member.collection_disposition,
           execution.terminal_outcome, execution.run_id,
           execution.run_provenance, execution.run_deleted_at IS NOT NULL,
           item.task_ref, item.task_digest, item.origin, item.workflow_role,
           member.collected_at, selected.accepted_at
      FROM anchors AS anchor
      JOIN audit_finding_contributions AS contribution
        ON contribution.finding_id = anchor.finding_id
       AND contribution.audit_id = anchor.audit_id
       AND contribution.receipt_id = anchor.receipt_id
      JOIN audit_execution_items AS member
        ON member.audit_id = anchor.audit_id AND member.item_id = anchor.item_id
      JOIN audit_executions AS execution
        ON execution.execution_id = member.execution_id
       AND execution.audit_id = member.audit_id
      JOIN audit_items AS item
        ON item.item_id = member.item_id AND item.audit_id = member.audit_id
      LEFT JOIN audit_finding_assessments AS selected
        ON selected.finding_id = anchor.finding_id
       AND selected.receipt_id = anchor.receipt_id
       AND selected.execution_item_id = member.execution_item_id
     WHERE anchor.audit_id = $1 AND anchor.finding_id = $2
    UNION ALL
    SELECT assessment.accepted_at, 'assessment:' || assessment.assessment_id,
           'direct-verification'::text,
           assessment.receipt_id, ''::text, contribution.proposal_ref,
           assessment.assessment_id, assessment.semantic_assessment,
           assessment.result_ref, assessment.result_digest,
           assessment.item_id, assessment.execution_item_id,
           assessment.collection_receipt_id, assessment.direct_verification,
           assessment.contract_ref, assessment.contract_digest,
           NULL::text, NULL::text, NULL::integer, NULL::text,
           NULL::text, NULL::text, NULL::text, NULL::jsonb,
           false, NULL::jsonb, NULL::text, NULL::jsonb, NULL::text,
           NULL::timestamptz, assessment.accepted_at
      FROM audit_finding_assessments AS assessment
      JOIN audit_finding_contributions AS contribution
        ON contribution.finding_id = assessment.finding_id
       AND contribution.receipt_id = assessment.receipt_id
     WHERE assessment.audit_id = $1 AND assessment.finding_id = $2
       AND assessment.direct_verification
)
SELECT created_at, record_id, kind, receipt_id, relation, proposal_ref,
       assessment_id, semantic_assessment, result_ref, result_digest,
       item_id, execution_item_id, collection_receipt_id, direct_verification,
       contract_ref, contract_digest, execution_id, execution_role,
       item_attempt, item_state, collection_disposition, terminal_outcome,
       run_id, run_provenance, run_deleted, task_ref, task_digest,
       item_origin, workflow_role, collected_at, assessment_accepted_at
  FROM records
 WHERE ($3::timestamptz IS NULL OR (created_at, record_id) > ($3, $4))
 ORDER BY created_at, record_id LIMIT $5`

// duplicateTargetCycleSQL reports whether the proposed duplicate target
// exists and whether following duplicate_target_id from it leads back to
// the finding being decided, which would close a cycle.
var duplicateTargetCycleSQL = `
WITH RECURSIVE chain(finding_id, duplicate_target_id) AS (
    SELECT finding_id, duplicate_target_id
      FROM audit_findings WHERE audit_id = $1 AND finding_id = $2
    UNION
    SELECT next.finding_id, next.duplicate_target_id
      FROM chain JOIN audit_findings AS next
        ON next.audit_id = $1 AND next.finding_id = chain.duplicate_target_id
     WHERE chain.duplicate_target_id IS NOT NULL
)
SELECT EXISTS (SELECT 1 FROM audit_findings WHERE audit_id = $1 AND finding_id = $2),
       EXISTS (SELECT 1 FROM chain WHERE finding_id = $3)`
