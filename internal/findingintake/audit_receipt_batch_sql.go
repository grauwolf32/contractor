package findingintake

// SQL statements for audit_receipt_batch.go.

// readOwnerAuditHoldsSQL reads the hold of each (Audit $1[i], receipt $2[i])
// pair, when that Audit belongs to owner $3 and to the hold's Project, ordered
// by receipt, creation and Audit. Used by Service.readAuditHoldsBatch.
var readOwnerAuditHoldsSQL = `
SELECT hold.receipt_id, hold.audit_id, hold.project_id, hold.proposal_ref,
       hold.evidence, hold.created_at
  FROM unnest($1::text[], $2::text[]) AS request(audit_id, receipt_id)
  JOIN finding_proposal_audit_holds AS hold
    ON hold.audit_id = request.audit_id AND hold.receipt_id = request.receipt_id
  JOIN audits AS audit
    ON audit.audit_id = hold.audit_id AND audit.project_id = hold.project_id
 WHERE audit.owner_id = $3
 ORDER BY hold.receipt_id, hold.created_at, hold.audit_id`
