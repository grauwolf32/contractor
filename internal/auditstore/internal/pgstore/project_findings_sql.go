package pgstore

const qualifiedProjectFindingColumns = `finding.finding_id,finding.project_id,finding.first_receipt_id,finding.first_proposal_ref,finding.title,finding.state,finding.analyst_verdict,finding.analyst_severity,finding.rationale,finding.actor_id,finding.assessed_at,finding.duplicate_target_id,finding.revision,finding.created_at,finding.updated_at,finding.correlation_algorithm,finding.correlation_value,finding.correlation_group_key,finding.correlation_reason`

// projectFindingColumns is the canonical row in scanProjectFinding order.
const projectFindingColumns = `finding_id,project_id,first_receipt_id,first_proposal_ref,title,state,analyst_verdict,analyst_severity,rationale,actor_id,assessed_at,duplicate_target_id,revision,created_at,updated_at,correlation_algorithm,correlation_value,correlation_group_key,correlation_reason`

const getProjectFindingSQL = `SELECT ` + projectFindingColumns + ` FROM project_findings AS finding
WHERE finding.project_id=$2 AND finding.finding_id=$3 AND EXISTS(
 SELECT 1 FROM projects WHERE project_id=$2 AND owner_id=$1 AND kind='project')`

// getAuditProjectFindingSQL authorizes the Audit and selects the same Project
// row through a source/check association. It never constructs an Audit identity.
const auditFindingMembershipSQL = `
 FROM project_findings AS finding
 JOIN project_finding_audits AS membership ON membership.finding_id=finding.finding_id AND membership.project_id=finding.project_id
 JOIN audits AS audit ON audit.audit_id=membership.audit_id AND audit.project_id=membership.project_id`

const getAuditProjectFindingSQL = `SELECT ` + qualifiedProjectFindingColumns + auditFindingMembershipSQL + `
 WHERE audit.owner_id=$1 AND membership.audit_id=$2 AND finding.finding_id=$3`

// assessProjectFindingSQL enforces the one Project CAS. New source admission
// never updates these fields. All current fields remain null until this command.
const assessProjectFindingSQL = `UPDATE project_findings SET
state=CASE $5::text WHEN 'true_positive' THEN 'confirmed' WHEN 'false_positive' THEN 'rejected'
 WHEN 'duplicate' THEN 'duplicate' WHEN 'reopen' THEN 'proposed' WHEN 'needs_evidence' THEN 'needs-evidence' END,
analyst_verdict=$5,analyst_severity=$6,rationale=$7,actor_id=$1,assessed_at=clock_timestamp(),
duplicate_target_id=$8,revision=revision+1,updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond')
WHERE project_id=$2 AND finding_id=$3 AND revision=$4 RETURNING ` + projectFindingColumns

// projectDuplicateTargetSQL traverses only current explicit duplicate links
// within the same Project, bounded to 127 target records plus the edited Finding (128 total).
const projectDuplicateTargetSQL = `WITH RECURSIVE chain(finding_id,duplicate_target_id,depth) AS (
 SELECT finding_id,duplicate_target_id,1 FROM project_findings WHERE project_id=$1 AND finding_id=$2
 UNION ALL
 SELECT next.finding_id,next.duplicate_target_id,chain.depth+1 FROM chain
 JOIN project_findings AS next ON next.project_id=$1 AND next.finding_id=chain.duplicate_target_id
 WHERE chain.duplicate_target_id IS NOT NULL AND chain.depth<127 AND chain.finding_id<>$3
)
SELECT EXISTS(SELECT 1 FROM chain), EXISTS(SELECT 1 FROM chain WHERE finding_id=$3),
 EXISTS(SELECT 1 FROM chain WHERE depth=127 AND duplicate_target_id IS NOT NULL)`
