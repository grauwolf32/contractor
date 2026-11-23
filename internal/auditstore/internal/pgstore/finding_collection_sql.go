package pgstore

// Collection snapshots do not create operational records or expand sources.
const readCollectionFindingsSQL = `SELECT source.receipt_id,finding.project_id,finding.finding_id,finding.revision,
 finding.state,coalesce(finding.analyst_verdict,''),coalesce(finding.analyst_severity,''),coalesce(finding.duplicate_target_id,'')
 FROM project_finding_sources AS source JOIN project_findings AS finding USING(project_id,finding_id)
 JOIN projects AS project USING(project_id)
 WHERE project.owner_id=$1 AND source.receipt_id=ANY($2::text[]) ORDER BY source.receipt_id LIMIT $3`

const readCollectionReceiptIDsSQL = `WITH requested AS (
 SELECT requested.finding_id,requested.ordinality FROM unnest($3::text[]) WITH ORDINALITY AS requested(finding_id,ordinality)
), owned AS (
 SELECT finding.finding_id,finding.revision FROM project_findings AS finding
 JOIN projects AS project USING(project_id)
 WHERE project.owner_id=$1 AND project.lifecycle_state='active' AND finding.finding_id=ANY($3::text[])
 AND (($5='project' AND finding.project_id=$2) OR ($5='audit' AND EXISTS(
 SELECT 1 FROM project_finding_audits AS membership JOIN audits AS audit USING(audit_id)
 WHERE membership.finding_id=finding.finding_id AND membership.audit_id=$2 AND audit.owner_id=$1)))
)
SELECT 0 AS kind,requested.ordinality,owned.revision,NULL::text AS receipt_id
 FROM requested LEFT JOIN owned USING(finding_id)
UNION ALL
(SELECT 1,NULL,NULL,source.receipt_id FROM owned CROSS JOIN LATERAL (
 SELECT receipt_id FROM project_finding_sources WHERE finding_id=owned.finding_id ORDER BY receipt_id LIMIT $4
) AS source ORDER BY source.receipt_id LIMIT $4)`
