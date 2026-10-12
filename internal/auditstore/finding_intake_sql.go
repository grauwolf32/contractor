package auditstore

// SQL statements for finding_intake.go.

// linkDirectFindingAssessmentSQL links two Artifacts to Audit $1: the direct
// verification result (logical key $2, ref $3, digest $4, media type $5, size
// $6, provenance $7, display ref $8) and its contract ($9-$15).
// Used by PostgresTxStore.CommitDirectFindingAssessment.
var linkDirectFindingAssessmentSQL = `
INSERT INTO audit_artifact_links (
    audit_id, logical_key, artifact_ref, artifact_digest,
    media_type, size_bytes, source_provenance, display_ref
) VALUES
    ($1, $2, $3::jsonb, $4, $5, $6, $7::jsonb, $8),
    ($1, $9, $10::jsonb, $11, $12, $13, $14::jsonb, $15)`

// insertDirectFindingAssessmentSQL inserts direct-verification assessment $1
// (semantic assessment $2, result $3/$4, contract $5/$6) for the finding that
// receipt $8 contributed to in Audit $7 and returns that finding's ID. No row
// means the receipt has no contribution in the Audit.
// Used by PostgresTxStore.CommitDirectFindingAssessment.
var insertDirectFindingAssessmentSQL = `
INSERT INTO audit_finding_assessments (
    assessment_id, finding_id, audit_id, receipt_id,
    semantic_assessment, result_ref, result_digest,
    direct_verification, contract_ref, contract_digest
)
SELECT $1, contribution.finding_id, contribution.audit_id, contribution.receipt_id,
       $2, $3::jsonb, $4, true, $5::jsonb, $6
  FROM audit_finding_contributions AS contribution
 WHERE contribution.audit_id = $7 AND contribution.receipt_id = $8
RETURNING finding_id`

// assessFindingDirectlySQL makes assessment $1 the current assessment of
// finding $3 in Audit $2: the finding returns to proposed without a decision,
// rejection reason or duplicate target, and its revision advances. Returns
// the new revision. Used by PostgresTxStore.CommitDirectFindingAssessment.
var assessFindingDirectlySQL = `
UPDATE audit_findings
   SET current_assessment_id = $1, current_decision_id = NULL,
       state = 'proposed', rejection_reason = NULL, duplicate_target_id = NULL,
       revision = revision + 1,
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id = $2 AND finding_id = $3
RETURNING revision`
