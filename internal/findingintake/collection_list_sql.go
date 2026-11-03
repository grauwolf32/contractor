package findingintake

// listPendingAuditReceiptsSQL walks one source Run's existing receipt index.
// Existing holds and rejection events exclude completed work before LIMIT.
// Used by Service.ListAuditCollection for bounded common Task collection.
var listPendingAuditReceiptsSQL = `
SELECT ` + receiptProjection + `
FROM finding_proposal_receipts AS receipt
JOIN workflow_runs AS run ON run.run_id=receipt.run_id AND run.owner_id=$1
JOIN audits AS audit ON audit.audit_id=$2 AND audit.owner_id=$1 AND audit.project_id=run.project_id
WHERE receipt.owner_id=$1 AND receipt.run_id=$3 AND receipt.audit_id=$2
 AND ($4::timestamptz IS NULL OR (receipt.created_at,receipt.receipt_id)>($4,$5))
 AND NOT EXISTS (SELECT 1 FROM finding_proposal_audit_holds AS held
   WHERE held.audit_id=$2 AND held.receipt_id=receipt.receipt_id)
 AND NOT EXISTS (SELECT 1 FROM audit_events AS rejected
   WHERE rejected.audit_id=$2 AND rejected.kind='finding.proposal_rejected' AND rejected.entity_id=receipt.receipt_id)
ORDER BY receipt.created_at,receipt.receipt_id LIMIT $6`
