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
     WHERE audit.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
       AND audit.revision = $4 AND audit.state = 'finalizing'
       AND $5 = (
           SELECT count(*) FROM audit_report_pages AS recorded
            WHERE recorded.audit_id = $1 AND recorded.audit_revision = $4
       )
       FOR SHARE OF audit, claim
), stale AS (
    DELETE FROM audit_report_pages AS page
     USING gate
     WHERE page.audit_id = gate.audit_id AND page.audit_revision <> $4
)
INSERT INTO audit_report_pages (
    audit_id, audit_revision, ordinal, section, round_id, last_item_ordinal,
    last_created_at, last_finding_id, exhausted, entry_count, content, coverage_counts
)
SELECT gate.audit_id, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
  FROM gate
ON CONFLICT (audit_id, audit_revision, ordinal) DO NOTHING
RETURNING ordinal`

// listReportPagesSQL lists the metadata of the report pages of Audit $1 at
// revision $2 in page order, without their content. $3 is one more than the
// page bound so the caller can detect overflow. Used by
// QueryStore.ListReportPages.
var listReportPagesSQL = `
SELECT ordinal, section, round_id, last_item_ordinal, last_created_at,
       last_finding_id, exhausted, entry_count, octet_length(content), coverage_counts
  FROM audit_report_pages
 WHERE audit_id = $1 AND audit_revision = $2
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
 WHERE audit_id = $1 AND audit_revision = $2
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
 ORDER BY item.ordinal
 LIMIT $4`

// listReportFindingsSQL pages the findings of Audit $1 in state $2 in
// creation order after the optional cursor ($3, $4), at most $5 rows, each
// with the report projection of its first receipt and its current
// assessment and decision. A finding is included while the projections
// before it total less than $6 bytes, so the first is always included;
// octet_length of the text projection reads only its stored length. Later
// rows come without a projection and only mark that the page is full. Used
// by QueryStore.ListReportFindings.
var listReportFindingsSQL = `
SELECT page.finding_id, page.state, page.first_proposal_ref,
       page.duplicate_target_id, page.revision, page.created_at, page.included,
       CASE WHEN page.included THEN page.report_projection END,
       assessment.assessment_id, assessment.semantic_assessment,
       assessment.result_ref, assessment.result_digest,
       assessment.direct_verification, assessment.contract_ref,
       assessment.contract_digest, assessment.accepted_at,
       decision.decision_id, decision.actor_id, decision.verdict,
       decision.severity, decision.rationale, decision.subject_revision,
       decision.subject_digest, decision.created_at
  FROM (
    SELECT finding.finding_id, finding.state, finding.first_proposal_ref,
           finding.duplicate_target_id, finding.revision, finding.created_at,
           finding.current_assessment_id, finding.current_decision_id,
           receipt.report_projection,
           sum(octet_length(receipt.report_projection)) OVER (
               ORDER BY finding.created_at, finding.finding_id
               ROWS UNBOUNDED PRECEDING
           ) - octet_length(receipt.report_projection) < $6 AS included
      FROM audit_findings AS finding
      JOIN finding_proposal_receipts AS receipt
        ON receipt.receipt_id = finding.first_receipt_id
     WHERE finding.audit_id = $1 AND finding.state = $2
       AND ($3::timestamptz IS NULL OR (finding.created_at, finding.finding_id) > ($3, $4))
     ORDER BY finding.created_at, finding.finding_id
     LIMIT $5
  ) AS page
  LEFT JOIN audit_finding_assessments AS assessment
    ON assessment.assessment_id = page.current_assessment_id
  LEFT JOIN audit_review_decisions AS decision
    ON decision.decision_id = page.current_decision_id
 ORDER BY page.created_at, page.finding_id`
