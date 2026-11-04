package pgstore

// purgeAuditDomainSQL clears derived back-references before deleting their
// immutable targets, then removes leaf rows before parents. The caller shares
// one remaining-row budget across all statements and commits each pass. Empty
// tables use their Audit FK indexes; pointer resets use partial purge indexes.
// purgeAuditItemsSQL selects a bounded ordered prefix by primary-row and
// distinct exact-hold counter cost. An unfit next row consumes the remainder
// of this pass, so later parent deletes cannot cascade past the item fence.
// A typed row conservatively reserves one staging-counter write even after
// acceptance; this avoids another parent lookup during bounded cleanup.
const purgeAuditItemsSQL = `
WITH candidates AS MATERIALIZED (
 SELECT ctid,item_id,1+retained_dependency_hold_count+
  CASE WHEN prepared_page_ordinal IS NULL THEN 0 ELSE 1 END AS work
 FROM audit_items WHERE audit_id=$1 ORDER BY item_id LIMIT $2
), weighted AS MATERIALIZED (
 SELECT *,sum(work) OVER (ORDER BY item_id) AS cumulative FROM candidates
), deleted AS (
 DELETE FROM audit_items AS target USING weighted AS batch
 WHERE target.ctid=batch.ctid AND batch.cumulative<=$2
 RETURNING batch.work
)
SELECT CASE WHEN EXISTS(SELECT 1 FROM weighted WHERE cumulative>$2) THEN $2
 ELSE COALESCE((SELECT sum(work) FROM deleted),0)::integer END`

var purgeAuditDomainSQL = []string{
	`WITH batch AS (SELECT audit_id,finding_id FROM project_finding_audits WHERE audit_id=$1 ORDER BY finding_id LIMIT LEAST($2,128))
 DELETE FROM project_finding_audits AS membership USING batch WHERE membership.audit_id=batch.audit_id AND membership.finding_id=batch.finding_id`,
	`WITH batch AS (
  SELECT ctid FROM audit_round_result_pages WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_round_result_pages AS target USING batch WHERE target.ctid = batch.ctid`,

	`WITH batch AS (
  SELECT ctid FROM audit_items WHERE audit_id = $1 AND (last_execution_item_id IS NOT NULL) LIMIT $2
)
UPDATE audit_items AS target SET last_execution_item_id = NULL
FROM batch WHERE target.ctid = batch.ctid`,

	`WITH batch AS (
  SELECT ctid FROM audit_execution_items WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_execution_items AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_tasks WHERE audit_id = $1 ORDER BY ordinal LIMIT $2
)
DELETE FROM audit_tasks AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (SELECT hypothesis_id,revision FROM audit_hypothesis_revisions WHERE audit_id=$1 ORDER BY hypothesis_id,revision LIMIT $2)
DELETE FROM audit_hypothesis_revisions r USING batch b WHERE r.hypothesis_id=b.hypothesis_id AND r.revision=b.revision`,
	`WITH batch AS (SELECT hypothesis_id FROM audit_hypotheses WHERE audit_id=$1 ORDER BY ordinal LIMIT $2)
DELETE FROM audit_hypotheses h USING batch b WHERE h.hypothesis_id=b.hypothesis_id`,
	purgeAuditItemsSQL,
	`WITH batch AS (
  SELECT ctid FROM audit_executions WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_executions AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_rounds WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_rounds AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_artifact_links WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_artifact_links AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_report_pages WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_report_pages AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_events WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_events AS target USING batch WHERE target.ctid = batch.ctid`,
}
