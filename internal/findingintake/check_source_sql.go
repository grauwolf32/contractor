package findingintake

// checkSourceMembershipSQL uses the canonical source index; a receipt from
// another Finding in the same Project does not authorize a check.
const checkSourceMembershipSQL = `SELECT EXISTS(
 SELECT 1 FROM project_finding_sources AS source
 JOIN finding_proposal_receipts AS receipt USING(receipt_id)
 WHERE receipt.owner_id=$1 AND source.project_id=$2 AND source.finding_id=$3
 AND source.receipt_id=$4 AND receipt.project_proposal IS NOT NULL)`
