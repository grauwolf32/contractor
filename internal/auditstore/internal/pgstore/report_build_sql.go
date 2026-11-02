package pgstore

// SQL statements for report_build.go.

// recordReportPageSQL appends page $5 of the report build of Audit $1 at
// revision $4 under the live claim ($1-$3). The Audit must be finalizing at
// that revision and $5 must follow the pages already recorded for it; the
// row locks keep a concurrent transition from discarding the build before
// this page commits. Pages of every other revision are stale and deleted.
// Returns no row when a precondition fails. Used by
// ClaimStore.RecordReportPage.
var recordReportPageSQL = `
WITH gate AS MATERIALIZED (
    SELECT audit.audit_id
      FROM audits AS audit
      JOIN audit_controller_claims AS claim USING (audit_id)
      JOIN finding_correlation_fences AS fence USING(project_id)
     WHERE audit.audit_id = $1 AND ` + auditClaims.LiveCondition("claim", "$2", "$3") + `
       AND audit.revision = $4 AND audit.state = 'finalizing' AND fence.revision=$15
       AND $5 = (
           SELECT count(*) FROM audit_report_pages AS recorded
            WHERE recorded.audit_id = $1 AND recorded.audit_revision = $4 AND recorded.project_revision=$15
       )
       FOR SHARE OF audit, claim, fence
), stale AS (
    DELETE FROM audit_report_pages AS page
     USING gate
     WHERE page.audit_id = gate.audit_id AND (page.audit_revision <> $4 OR page.project_revision<>$15)
)
INSERT INTO audit_report_pages (
    audit_id, audit_revision, project_revision, ordinal, section, round_id, last_item_ordinal,
    last_created_at, last_finding_id, exhausted, entry_count, content, coverage_counts
)
SELECT gate.audit_id, $4, $15, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
  FROM gate
ON CONFLICT (audit_id, audit_revision, project_revision, ordinal) DO NOTHING
RETURNING ordinal`

// listReportPagesSQL lists the metadata of the report pages of Audit $1 at
// revision $2 in page order, without their content. $3 is one more than the
// page bound so the caller can detect overflow. Used by
// QueryStore.ListReportPages.
var listReportPagesSQL = `
SELECT ordinal, section, round_id, last_item_ordinal, last_created_at,
       last_finding_id, exhausted, entry_count, octet_length(content), coverage_counts
  FROM audit_report_pages
 WHERE audit_id = $1 AND audit_revision = $2 AND project_revision=$4
 ORDER BY ordinal
 LIMIT $3`

// readReportSectionsSQL joins the recorded entries of each report section of
// Audit $1 at revision $2 with commas, in page order, and counts them. Used
// by QueryStore.ReadReportSections.
var readReportSectionsSQL = `
SELECT section, sum(entry_count)::integer,
       COALESCE(string_agg(content, '\x2c'::bytea ORDER BY ordinal)
                FILTER (WHERE entry_count > 0), '\x'::bytea)
  FROM audit_report_pages
 WHERE audit_id = $1 AND audit_revision = $2 AND project_revision=$3
 GROUP BY section`

// listReportItemsSQL pages the items of round $2 in Audit $1 with their
// coverage, in item ordinal order after $3 and limited to $4 rows, over the
// (round_id, ordinal) key. Used by QueryStore.ListReportItems.
var listReportItemsSQL = `
SELECT item.item_id, item.round_id, item.item_key, item.ordinal, item.kind,
       item.subject_key, item.task_ref, item.task_digest, item.state,
       item.final_disposition, item.coverage_status, item.coverage_requested,
       item.coverage_completed, item.coverage_gaps, item.coverage_rationale,
       item.coverage_result_ref, item.coverage_result_digest
  FROM audit_items AS item
 WHERE item.audit_id = $1 AND item.round_id = $2 AND item.ordinal > $3
   AND ` + acceptedItemSQL("item") + `
 ORDER BY item.ordinal
 LIMIT $4`

// Bound the indexed membership window before reading projection bytes or
// computing a running byte sum. NULL current assessments share the possible
// issues section with reopened Findings; the actual state stays NULL.
const listReportFindingsSQL = `WITH candidates AS MATERIALIZED (
 SELECT membership.* FROM project_finding_audits AS membership
 WHERE membership.audit_id=$1
 AND (membership.finding_created_at,membership.finding_id COLLATE "C")>(coalesce($3::timestamptz,'-infinity'),coalesce($4::text,'') COLLATE "C")
 ORDER BY membership.finding_created_at,membership.finding_id COLLATE "C" LIMIT $5
), selected AS MATERIALIZED (
 SELECT finding.*,(finding.state=$2 OR (finding.state IS NULL AND $2='proposed')) IS TRUE AS matches
 FROM candidates CROSS JOIN LATERAL (SELECT * FROM project_findings AS current
 WHERE current.project_id=candidates.project_id AND current.finding_id=candidates.finding_id OFFSET 0) AS finding
), page AS MATERIALIZED (
 SELECT selected.*,CASE WHEN matches THEN receipt.report_projection END AS report_projection,
 sum(CASE WHEN matches THEN octet_length(receipt.report_projection) ELSE 0 END)
 OVER(ORDER BY selected.created_at,selected.finding_id COLLATE "C" ROWS UNBOUNDED PRECEDING)
 -CASE WHEN matches THEN octet_length(receipt.report_projection) ELSE 0 END<$6 AS included
 FROM selected CROSS JOIN LATERAL (SELECT report_projection FROM finding_proposal_receipts AS source
 WHERE source.receipt_id=selected.first_receipt_id OFFSET 0) AS receipt
)
 SELECT finding_id,project_id,state,first_proposal_ref,duplicate_target_id,revision,created_at,included,matches,
 CASE WHEN included THEN report_projection END,analyst_verdict,analyst_severity,rationale,actor_id,assessed_at
 FROM page ORDER BY created_at,finding_id COLLATE "C"`

const reportProjectRevisionSQL = `SELECT fence.revision FROM audits AS audit
 JOIN finding_correlation_fences AS fence USING(project_id) WHERE audit.audit_id=$1`
