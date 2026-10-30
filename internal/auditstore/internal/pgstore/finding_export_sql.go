package pgstore

// findingExportScopeSQL authorizes one Audit of $1 at $2, excluding deletion.
// FindingExportScope adds FOR SHARE only for publication, fencing lifecycle.
const findingExportScopeSQL = `SELECT project_id FROM audits WHERE owner_id=$1 AND audit_id=$2 AND state <> 'deleting'`

// exportFindingsSQL reads at most the bounded explicit IDs $3 in owner $1's
// Audit $2, in the existing validated finding-record column order.
var exportFindingsSQL = `SELECT ` + findingRecordColumns + `
 FROM audit_findings AS finding JOIN audits AS audit USING (audit_id)
 WHERE audit.owner_id=$1 AND audit.audit_id=$2 AND finding.finding_id=ANY($3::text[])
 ORDER BY finding.finding_id`

// exportContributionsSQL walks only the explicit indexed finding IDs $2 of
// Audit $1. Each finding probe stops at sentinel $3; no whole-Audit scan.
const exportContributionsSQL = `
SELECT requested.finding_id, contribution.receipt_id
  FROM unnest($2::text[]) AS requested(finding_id)
 CROSS JOIN LATERAL (
    SELECT receipt_id FROM audit_finding_contributions
     WHERE audit_id=$1 AND finding_id=requested.finding_id
     ORDER BY receipt_id LIMIT $3
 ) AS contribution
 ORDER BY requested.finding_id, contribution.receipt_id`
