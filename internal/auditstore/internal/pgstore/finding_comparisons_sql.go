package pgstore

// projectCorrelationFenceJoin joins the finding correlation fence row of the
// Project aliased project, and projectCorrelationRevision is that Project's
// opaque canonical Finding revision. Every ordinary Project owns its fence
// from creation; a missing fence is invalid and never supplies a default.
const (
	projectCorrelationFenceJoin = `
 JOIN finding_correlation_fences AS fence ON fence.project_id=project.project_id`
	projectCorrelationRevision = `fence.revision`
)

// correlationFenceSQL reads only the two selected Audit identities and compact
// baseline identities. No profile, evidence or whole-Audit inventory is hydrated.
const correlationFenceSQL = `SELECT ` + projectCorrelationRevision + `,
 audit.audit_id,audit.revision,audit.state,audit.profile_name,audit.profile_version,audit.profile_digest,
 CASE WHEN audit.baseline_snapshot IS NULL THEN NULL ELSE 'sha256:'||encode(sha256(convert_to(audit.baseline_snapshot::text,'UTF8')),'hex') END
 FROM projects AS project` + projectCorrelationFenceJoin + `
 LEFT JOIN audits AS audit ON audit.project_id=project.project_id
 AND audit.owner_id=$1 AND audit.audit_id=ANY($3::text[])
 WHERE project.owner_id=$1 AND project.project_id=$2 AND project.kind='project' AND project.lifecycle_state='active' ORDER BY audit.audit_id COLLATE "C"`

// CandidateSource authorizes the selected source through the canonical Finding
// and the explicit Audit membership. Source receipt metadata is immutable.
const candidateSourceSQL = `SELECT finding.finding_id,audit.audit_id,finding.project_id,
 finding.first_receipt_id,receipt.receipt_id,finding.correlation_group_key,receipt.comparison_projection,
 audit.revision,fence.revision
 FROM project_findings AS finding
 JOIN project_finding_audits AS membership USING(project_id,finding_id)
 JOIN audits AS audit USING(audit_id)
 JOIN finding_correlation_fences AS fence ON fence.project_id=finding.project_id
 JOIN project_finding_sources AS source ON source.finding_id=finding.finding_id AND source.receipt_id=coalesce($4::text,finding.first_receipt_id)
 JOIN finding_proposal_receipts AS receipt USING(receipt_id)
 WHERE audit.owner_id=$1 AND membership.audit_id=$2 AND finding.finding_id=$3`

const candidateWindowColumns = `SELECT finding.finding_id,''::text,finding.first_receipt_id,
 finding.correlation_group_key,finding.created_at,receipt.comparison_projection-'preview'
 FROM selected AS finding JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id=finding.first_receipt_id
 ORDER BY finding.created_at,finding.finding_id COLLATE "C"`

const candidateProjectSelectionSQL = `SELECT finding_id,first_receipt_id,correlation_group_key,created_at FROM project_findings
 WHERE project_id=$1 AND finding_id<>$2 AND (created_at,finding_id COLLATE "C")>($3,$4)`
const candidateProjectWindowSQL = `WITH selected AS MATERIALIZED (` + candidateProjectSelectionSQL + `
 ORDER BY created_at,finding_id COLLATE "C" LIMIT $5) ` + candidateWindowColumns
const candidateProjectGroupWindowSQL = `WITH selected AS MATERIALIZED (` + candidateProjectSelectionSQL + `
 AND correlation_group_key=$6 AND correlation_group_key<>'' ORDER BY created_at,finding_id COLLATE "C" LIMIT $5) ` + candidateWindowColumns

const candidateAuditSelectionSQL = `SELECT finding.finding_id,finding.first_receipt_id,finding.correlation_group_key,finding.created_at
 FROM project_finding_audits AS membership JOIN project_findings AS finding USING(project_id,finding_id)
 WHERE membership.audit_id=$1 AND membership.finding_id<>$2
 AND (membership.finding_created_at,membership.finding_id COLLATE "C")>($3,$4)`
const candidateAuditWindowSQL = `WITH selected AS MATERIALIZED (` + candidateAuditSelectionSQL + `
 ORDER BY membership.finding_created_at,membership.finding_id COLLATE "C" LIMIT $5) ` + candidateWindowColumns
const candidateAuditGroupWindowSQL = `WITH selected AS MATERIALIZED (` + candidateAuditSelectionSQL + `
 AND membership.correlation_group_key COLLATE "C"=$6 AND membership.correlation_group_key<>''
 ORDER BY membership.finding_created_at,membership.finding_id COLLATE "C" LIMIT $5) ` + candidateWindowColumns

// Each side seeks at most limit+1 memberships, regardless of source count.
const comparisonGroupsSQL = `WITH keys AS MATERIALIZED (
 SELECT correlation_group_key FROM (
 (SELECT correlation_group_key COLLATE "C" FROM project_finding_audits WHERE audit_id=$1 AND correlation_group_key<>''
 AND correlation_group_key COLLATE "C">$3 ORDER BY correlation_group_key COLLATE "C" LIMIT $4)
 UNION
 (SELECT correlation_group_key COLLATE "C" FROM project_finding_audits WHERE audit_id=$2 AND correlation_group_key<>''
 AND correlation_group_key COLLATE "C">$3 ORDER BY correlation_group_key COLLATE "C" LIMIT $4)
 ) AS both_keys ORDER BY correlation_group_key COLLATE "C" LIMIT $4
)
 SELECT keys.correlation_group_key,finding.correlation_algorithm,finding.correlation_value,
 l.finding_id,r.finding_id,''::text
 FROM keys
 LEFT JOIN project_finding_audits AS l ON l.audit_id=$1 AND l.correlation_group_key<>'' AND l.correlation_group_key COLLATE "C"=keys.correlation_group_key
 LEFT JOIN project_finding_audits AS r ON r.audit_id=$2 AND r.correlation_group_key<>'' AND r.correlation_group_key COLLATE "C"=keys.correlation_group_key
 JOIN project_findings AS finding ON finding.finding_id=coalesce(l.finding_id,r.finding_id)
 ORDER BY keys.correlation_group_key COLLATE "C"`

const comparisonUnmatchedSQL = `WITH keys AS MATERIALIZED (
 SELECT finding_id FROM (
 (SELECT finding_id COLLATE "C" FROM project_finding_audits WHERE audit_id=$1 AND correlation_group_key=''
 AND finding_id COLLATE "C">$3 ORDER BY finding_id COLLATE "C" LIMIT $4)
 UNION
 (SELECT finding_id COLLATE "C" FROM project_finding_audits WHERE audit_id=$2 AND correlation_group_key=''
 AND finding_id COLLATE "C">$3 ORDER BY finding_id COLLATE "C" LIMIT $4)
 ) AS both_keys ORDER BY finding_id COLLATE "C" LIMIT $4
)
 SELECT ''::text,''::text,''::text,l.finding_id,r.finding_id,finding.correlation_reason
 FROM keys JOIN project_findings AS finding USING(finding_id)
 LEFT JOIN project_finding_audits AS l ON l.audit_id=$1 AND l.finding_id=finding.finding_id
 LEFT JOIN project_finding_audits AS r ON r.audit_id=$2 AND r.finding_id=finding.finding_id
 ORDER BY keys.finding_id COLLATE "C"`

const comparisonPreviewsSQL = `SELECT finding.finding_id,finding.project_id,finding.revision,finding.state,
 receipt.comparison_projection->'preview',finding.analyst_verdict,finding.analyst_severity,
 receipt.receipt_id,receipt.run_id,receipt.source_run_deleted_at IS NOT NULL,receipt.logical_agent_name,
 receipt.workflow_name,receipt.workflow_version,receipt.workflow_closure_digest
 FROM project_findings AS finding JOIN projects AS project USING(project_id)
 JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id=finding.first_receipt_id
 WHERE project.owner_id=$1 AND finding.project_id=$2 AND finding.finding_id=ANY($3::text[])`

// checkCorrelationFenceSQL rechecks only scalar revisions after hydration; the
// immutable baseline/profile metadata need not be read or hashed a second time.
const checkCorrelationFenceSQL = `SELECT
 EXISTS(SELECT 1 FROM projects AS project` + projectCorrelationFenceJoin + `
 WHERE project.owner_id=$1 AND project.project_id=$2
 AND ($3::bigint=0 OR ` + projectCorrelationRevision + `=$3))
 AND (SELECT count(*) FROM unnest($4::text[],$5::bigint[]) AS expected(audit_id,revision)
 JOIN audits AS audit ON audit.audit_id=expected.audit_id AND audit.revision=expected.revision
 WHERE audit.owner_id=$1 AND audit.project_id=$2)=cardinality($4::text[])`
