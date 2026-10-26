package findingintake

// pendingChecksProjectionSQL reads only small exact source headers for the
// next bounded data-migration batch. A retained Project copy survives source
// Run deletion; the hold's descriptor must still match the original content.
const pendingChecksProjectionSQL = `
SELECT receipt.receipt_id,receipt.run_id,receipt.proposal_ref,receipt.proposal_digest,
 receipt.proposal_media_type,receipt.proposal_size_bytes,receipt.retention_state,
 held.project_id,held.proposal_ref
FROM finding_proposal_receipts AS receipt
LEFT JOIN LATERAL (
 SELECT hold.project_id,hold.proposal_ref
 FROM finding_proposal_audit_holds AS hold
 JOIN audits AS audit ON audit.audit_id=hold.audit_id AND audit.project_id=hold.project_id
 WHERE hold.receipt_id=receipt.receipt_id AND audit.owner_id=receipt.owner_id
 ORDER BY hold.audit_id LIMIT 1
) AS held ON true
WHERE receipt.checks_projection IS NULL
ORDER BY receipt.receipt_id LIMIT $1`

// writeChecksProjectionSQL is the one-time exact-source CAS. A concurrent
// discard has already erased the index and cannot be resurrected by this job.
const writeChecksProjectionSQL = `
UPDATE finding_proposal_receipts SET checks_projection=$2::text::json
WHERE receipt_id=$1 AND checks_projection IS NULL AND retention_state<>'discarded'
 AND proposal_ref=$3::jsonb AND proposal_digest=$4
 AND proposal_media_type=$5 AND proposal_size_bytes=$6`
