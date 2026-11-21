package pgstore

import (
	"strings"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

const projectFindingSummaryColumns = `finding.finding_id,finding.project_id,left(finding.title,256),length(finding.title)>256,
finding.subject_kind,finding.subject_key,finding.subject_truncated,finding.severity_suggestion,
finding.state,finding.analyst_verdict,finding.analyst_severity,finding.duplicate_target_id,
finding.revision,finding.created_at,finding.updated_at`

const findingPredicatesSQL = "\n/* finding predicates */"

// Select closed predicates before preparing a query. Generic plans cannot
// estimate optional-OR filters and otherwise scan/sort an entire Project even
// when all filters are absent. The unused CTE preserves parameter types.
func findingPageSQL(query string, state *auditstore.FindingState, verdict *auditstore.AnalystVerdict, needsReview bool, severity *auditstore.FindingSeverity) string {
	filters := ""
	if state != nil {
		filters += " AND finding.state=$3::text"
	}
	if verdict != nil {
		filters += " AND finding.analyst_verdict=$4::text"
	}
	if needsReview {
		filters += " AND (finding.analyst_verdict IS NULL OR finding.state='proposed') AND coalesce(finding.analyst_severity,finding.severity_suggestion,'')<>'informational'"
	}
	if severity != nil {
		filters += " AND coalesce(finding.analyst_severity,finding.severity_suggestion)= $6::text"
	}
	if (needsReview && verdict != nil) ||
		(verdict != nil && state != nil && (*verdict == auditstore.VerdictTruePositive && *state != auditstore.FindingConfirmed || *verdict == auditstore.VerdictFalsePositive && *state != auditstore.FindingRejected)) {
		filters = " AND false"
	}

	return `WITH finding_filter_parameters AS (SELECT $3::text,$4::text,$5::boolean,$6::text) ` + strings.Replace(query, findingPredicatesSQL, filters, 1)
}

// Limit membership candidates before reading canonical state. Filtering can
// yield an empty progress page; no join may search an entire Audit to fill it.
var listFindingsSQL = `SELECT ` + projectFindingSummaryColumns + `
 FROM (
   SELECT membership.* FROM project_finding_audits AS membership
   JOIN audits AS audit USING(audit_id)
   WHERE audit.owner_id=$1 AND membership.audit_id=$2
   AND (membership.finding_created_at,membership.finding_id COLLATE "C")>(coalesce($7::timestamptz,'-infinity'),$8::text COLLATE "C")
   ORDER BY membership.finding_created_at,membership.finding_id COLLATE "C" LIMIT $9
 ) AS membership
 CROSS JOIN LATERAL (SELECT * FROM project_findings AS current
   WHERE current.project_id=membership.project_id AND current.finding_id=membership.finding_id OFFSET 0) AS finding
 WHERE true` + findingPredicatesSQL + `
 ORDER BY membership.finding_created_at,membership.finding_id COLLATE "C"`

const findingPageBasisSQL = `SELECT audit.revision,fence.revision,statement_timestamp()
 FROM audits AS audit JOIN finding_correlation_fences AS fence USING(project_id)
 WHERE audit.owner_id=$1 AND audit.audit_id=$2`

// owner_id is an immutable index scope derived from the Project at creation.
// First bound candidates within this owner; then check Project visibility.
// Deleting/evaluation Projects cannot force a full owner-history scan.
var listOwnerFindingsSQL = `WITH filter_parameters AS (SELECT $2::text,$3::text,$4::text,$5::boolean,$6::text)
 SELECT ` + projectFindingSummaryColumns + `, project.owner_id=$1 AND ` + projectstore.OwnerListedProjectSQL("project.project_id", "project.owner_id") + ` AS visible
 FROM (
   SELECT * FROM project_findings WHERE owner_id=$1
   AND (created_at,finding_id COLLATE "C")<(coalesce($7::timestamptz,'infinity'),$8::text COLLATE "C")
   ORDER BY created_at DESC,finding_id COLLATE "C" DESC LIMIT $9
 ) AS finding
 CROSS JOIN LATERAL (SELECT * FROM projects WHERE project_id=finding.project_id OFFSET 0) AS project
 ORDER BY finding.created_at DESC,finding.finding_id COLLATE "C" DESC`

var listProjectFindingsSQL = `WITH filter_parameters AS (SELECT $3::text,$4::text,$5::boolean,$6::text)
 SELECT ` + projectFindingSummaryColumns + `,true
 FROM project_findings AS finding JOIN projects AS project USING(project_id)
 WHERE project.owner_id=$1 AND finding.project_id=$2 AND project.lifecycle_state='active'
 AND (finding.created_at,finding.finding_id COLLATE "C")<(coalesce($7::timestamptz,'infinity'),$8::text COLLATE "C")
 ORDER BY finding.created_at DESC,finding.finding_id COLLATE "C" DESC LIMIT $9`
