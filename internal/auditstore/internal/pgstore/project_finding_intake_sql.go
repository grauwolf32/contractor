package pgstore

// admitProjectFindingSQL uses the existing qualified-key selection. The
// caller holds the Project fence, so a matching concurrent publication either
// precedes this transaction or receives the ordinary retryable conflict.
// A missing/ambiguous primary key never merges independent receipts.
const admitProjectFindingSQL = `
WITH qualified AS MATERIALIZED (
    SELECT project.owner_id, receipt.receipt_id, receipt.project_proposal, receipt.report_projection::jsonb->>'title' AS title,
           left(coalesce(receipt.report_projection::jsonb->'subject'->>'kind',''),64) AS subject_kind,
           left(coalesce(receipt.report_projection::jsonb->'subject'->>'key',''),256) AS subject_key,
           length(coalesce(receipt.report_projection::jsonb->'subject'->>'key',''))>256 OR
             length(coalesce(receipt.report_projection::jsonb->'subject'->>'kind',''))>64 AS subject_truncated,
           receipt.report_projection::jsonb->>'severity_suggestion' AS severity_suggestion, primary_key.*,
           CASE WHEN primary_key.algorithm='' THEN ''
                ELSE contractor_finding_group_key($1,primary_key.algorithm,primary_key.value) END AS group_key
      FROM finding_proposal_receipts AS receipt
      JOIN projects AS project ON project.project_id=receipt.project_id
      CROSS JOIN LATERAL contractor_finding_primary_key(receipt.receipt_id) AS primary_key
     WHERE receipt.project_id=$1 AND receipt.receipt_id=$2
       AND receipt.publication_requested AND receipt.publication_error=''
       AND receipt.project_proposal IS NOT NULL AND jsonb_array_length(receipt.project_evidence)>0
       AND project.kind='project' AND project.lifecycle_state<>'deleting'
), matched AS MATERIALIZED (
    SELECT finding.finding_id FROM project_findings AS finding JOIN qualified
        ON qualified.group_key<>'' AND finding.project_id=$1 AND finding.correlation_group_key=qualified.group_key
), inserted AS (
    INSERT INTO project_findings(finding_id,project_id,owner_id,first_receipt_id,first_proposal_ref,title,subject_kind,subject_key,subject_truncated,severity_suggestion,
        correlation_algorithm,correlation_value,correlation_group_key,correlation_reason)
    SELECT $3,$1,owner_id,receipt_id,project_proposal,title,subject_kind,subject_key,subject_truncated,severity_suggestion,algorithm,value,group_key,reason FROM qualified
     WHERE NOT EXISTS(SELECT 1 FROM matched)
    RETURNING finding_id
)
SELECT finding_id FROM matched UNION ALL SELECT finding_id FROM inserted`

// linkFindingSourceAuditSQL references the canonical identity through the
// receipt. Native provenance and explicit import use the same membership.
const linkFindingSourceAuditSQL = `INSERT INTO project_finding_audits(audit_id,finding_id,project_id,finding_created_at,correlation_group_key)
 SELECT audit.audit_id,finding.finding_id,finding.project_id,finding.created_at,finding.correlation_group_key
 FROM project_finding_sources AS source
 JOIN project_findings AS finding USING(project_id,finding_id)
 JOIN finding_proposal_receipts AS receipt USING(receipt_id)
 JOIN audits AS audit ON audit.audit_id=coalesce(nullif($2::text,''),receipt.audit_id) AND audit.project_id=finding.project_id
 WHERE source.receipt_id=$1 ON CONFLICT DO NOTHING`
