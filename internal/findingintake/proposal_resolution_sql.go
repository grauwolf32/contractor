package findingintake

// SQL statements for proposal_resolution.go.

// resolveAuditProposalsSQL resolves each (invocation $5[i], client key $6[i])
// selection of the Audit child Run $4 to the receipts that execution $3 of
// owner $1's Audit $2 retained: the receipt belongs to the owner, the Audit,
// the execution and the Run, and the Audit, which belongs to the owner, holds
// it. It returns at most two receipts per selection, so a selection that
// matches several is detectable, in selection order. A row holds the receipt,
// the selection's 1-based ordinal and the Audit's hold of the receipt.
// Used by Service.ResolveAuditProposals.
var resolveAuditProposalsSQL = `
WITH request AS (
    SELECT selection.invocation_id, selection.client_key, selection.ordinality
      FROM unnest($5::text[], $6::text[])
           WITH ORDINALITY AS selection(invocation_id, client_key, ordinality)
), matched AS (
    SELECT request.ordinality, receipt.receipt_id,
           row_number() OVER (PARTITION BY request.ordinality ORDER BY receipt.receipt_id) AS rank
      FROM request
      JOIN finding_proposal_receipts AS receipt
        ON receipt.run_id = $4 AND receipt.invocation_id = request.invocation_id
       AND receipt.client_key = request.client_key
      JOIN finding_proposal_audit_holds AS hold
        ON hold.receipt_id = receipt.receipt_id AND hold.audit_id = $2
      JOIN audits AS audit ON audit.audit_id = hold.audit_id
     WHERE audit.owner_id = $1 AND receipt.owner_id = $1
       AND receipt.audit_id = $2 AND receipt.audit_execution_id = $3
)
SELECT ` + receiptProjection + `, matched.ordinality,
       hold.audit_id, hold.project_id, hold.proposal_ref, hold.evidence, hold.created_at
  FROM matched
  JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id = matched.receipt_id
  JOIN finding_proposal_audit_holds AS hold
    ON hold.receipt_id = matched.receipt_id AND hold.audit_id = $2
 WHERE matched.rank <= 2
 ORDER BY matched.ordinality, matched.rank`
