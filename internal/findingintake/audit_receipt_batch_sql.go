package findingintake

// SQL statements for audit_receipt_batch.go.

// readOwnerAuditReceiptsSQL reads the receipts of the (Audit $2[i], receipt
// $3[i]) pairs that owner $1 may read through that Audit: the Audit belongs
// to the owner, and the receipt belongs to the owner and is either the
// Audit's native child receipt or held by the Audit in its own Project. Each
// row ends with the Audit the receipt was read through.
// Used by Service.GetOwnerAuditReceipts.
var readOwnerAuditReceiptsSQL = `
SELECT ` + receiptProjection + `, request.audit_id
  FROM unnest($2::text[], $3::text[]) AS request(audit_id, receipt_id)
  JOIN audits AS audit ON audit.audit_id = request.audit_id AND audit.owner_id = $1
  JOIN finding_proposal_receipts AS receipt
    ON receipt.receipt_id = request.receipt_id AND receipt.owner_id = $1
 WHERE receipt.audit_id = request.audit_id OR EXISTS (
       SELECT 1 FROM finding_proposal_audit_holds AS hold
        WHERE hold.receipt_id = receipt.receipt_id AND hold.audit_id = request.audit_id
          AND hold.project_id = audit.project_id
   )`

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
