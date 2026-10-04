package auditstore

// SQL statements for findings.go.

// listReportFindingsSQL reads every finding of Audit $1 with its current
// assessment and current review decision (both optional), in creation order.
// The 100001-row limit lets the caller reject projections over 100000
// findings. Used by PostgresStore.ListReportFindings.
var listReportFindingsSQL = `
SELECT finding.finding_id, finding.state, finding.first_proposal_ref,
       finding.duplicate_target_id, finding.revision,
       assessment.assessment_id, assessment.semantic_assessment,
       assessment.result_ref, assessment.result_digest,
       assessment.direct_verification, assessment.contract_ref,
       assessment.contract_digest, assessment.accepted_at,
       decision.decision_id, decision.actor_id, decision.verdict,
       decision.severity, decision.rationale, decision.subject_revision,
       decision.subject_digest, decision.created_at
  FROM audit_findings AS finding
  LEFT JOIN audit_finding_assessments AS assessment
    ON assessment.assessment_id = finding.current_assessment_id
  LEFT JOIN audit_review_decisions AS decision
    ON decision.decision_id = finding.current_decision_id
 WHERE finding.audit_id = $1
 ORDER BY finding.created_at, finding.finding_id
 LIMIT 100001`
