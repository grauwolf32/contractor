package findingintake

// auditExecutionReceiptReferencesSQL uses the execution source index and the
// existing hold key, never an Audit-wide scan or a proposal payload read.
// Used by Service.AuditExecutionReceiptReferences during common result assembly.
var auditExecutionReceiptReferencesSQL = `
SELECT receipt.audit_task_id,receipt.receipt_id,source.finding_id
FROM finding_proposal_receipts AS receipt
LEFT JOIN project_finding_sources AS source USING(receipt_id)
JOIN finding_proposal_audit_holds AS held
 ON held.receipt_id=receipt.receipt_id AND held.audit_id=$2
WHERE receipt.owner_id=$1 AND receipt.audit_id=$2
 AND receipt.audit_execution_id=$3 AND receipt.run_id=$4
ORDER BY receipt.audit_task_id,receipt.created_at,receipt.receipt_id
LIMIT $5`
