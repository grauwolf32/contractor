package findingintake

// SQL statements for direct_verification.go.

// commitDirectVerificationSQL inserts direct-verification assessment $1
// (semantic assessment $2, result $3/$4, contract $5/$6) for the finding that
// receipt $8 contributed to in Audit $7 and returns that finding's ID. No row
// means the receipt has no contribution in the Audit.
// Used by commitDirectVerification.
var commitDirectVerificationSQL = `
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
