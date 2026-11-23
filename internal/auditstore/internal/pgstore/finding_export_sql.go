package pgstore

// Audit exports authorize the selected filter but capture the canonical
// Project assessment and every source of each explicitly selected Finding.
const findingExportScopeSQL = `SELECT project_id FROM audits WHERE owner_id=$1 AND audit_id=$2 AND state<>'deleting'`

const exportFindingsSQL = `SELECT ` + qualifiedProjectFindingColumns + auditFindingMembershipSQL + `
 WHERE audit.owner_id=$1 AND membership.audit_id=$2 AND finding.finding_id=ANY($3::text[]) ORDER BY finding.finding_id`

const exportContributionsSQL = `SELECT requested.finding_id,source.receipt_id
 FROM unnest($2::text[]) AS requested(finding_id)
 JOIN project_finding_audits AS membership ON membership.finding_id=requested.finding_id AND membership.audit_id=$1
 CROSS JOIN LATERAL (SELECT receipt_id FROM project_finding_sources
 WHERE finding_id=requested.finding_id ORDER BY receipt_id LIMIT $3) AS source
 ORDER BY requested.finding_id,source.receipt_id`
