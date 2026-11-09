package findingintake

// readExecutionReceiptSQL reconciles the primary identity derived from trusted
// immutable candidate-version provenance, including in a new publication Run.
var readExecutionReceiptSQL = `SELECT ` + receiptProjection + `
FROM finding_proposal_receipts AS receipt WHERE receipt.receipt_id=$1`
