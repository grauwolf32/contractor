package findingintake

// SQL statements for postgres.go.

// importIntoAuditWithTxSQL locks owner $1's retained proposal receipt for Run
// $3 and exact proposal ref $4, its retention row and destination Audit $2 in
// the Run's Project. The Audit must be in one of states $5, require human
// finding confirmation and have no pending report review. Returns the receipt's
// proposal and evidence data. Used by Service.importIntoAuditWithTx.
var importIntoAuditWithTxSQL = `
SELECT receipt.receipt_id, audit.project_id, receipt.proposal_ref, receipt.evidence,
       receipt.invocation_id, receipt.client_key, receipt.workflow_closure_digest,
       receipt.proposal_digest, receipt.proposal_media_type, receipt.proposal_size_bytes
  FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_retention AS retention USING (receipt_id)
  JOIN workflow_runs AS run ON run.run_id = receipt.run_id
  JOIN audits AS audit ON audit.audit_id = $2
 WHERE receipt.owner_id = $1 AND receipt.run_id = $3
   AND receipt.proposal_ref = $4::jsonb
   AND run.owner_id = $1 AND run.project_id = audit.project_id
   AND audit.owner_id = $1
   AND audit.state = ANY($5::text[])
   AND NOT EXISTS (
       SELECT 1
         FROM audit_report_candidates AS candidate
         JOIN audit_review_requests AS review
           ON review.request_id = candidate.request_id
          AND review.audit_id = candidate.audit_id
        WHERE candidate.audit_id = audit.audit_id AND review.state = 'pending'
   )
   AND audit.profile_snapshot #>> '{interaction,findingConfirmation}' = 'human-required'
 FOR UPDATE OF receipt, retention, audit`
