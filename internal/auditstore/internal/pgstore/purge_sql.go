package pgstore

// purgeAuditDomainSQL clears derived back-references before deleting their
// immutable targets, then removes leaf rows before parents. The caller shares
// one remaining-row budget across all statements and commits each pass. Empty
// tables use their Audit FK indexes; pointer resets use partial purge indexes.
var purgeAuditDomainSQL = []string{
	`WITH batch AS (
  SELECT ctid FROM audit_findings WHERE audit_id = $1 AND (current_assessment_id IS NOT NULL OR current_decision_id IS NOT NULL OR duplicate_target_id IS NOT NULL) LIMIT $2
)
UPDATE audit_findings AS target SET state = 'proposed', rejection_reason = NULL, duplicate_target_id = NULL, current_assessment_id = NULL, current_decision_id = NULL
FROM batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_items WHERE audit_id = $1 AND (last_execution_item_id IS NOT NULL) LIMIT $2
)
UPDATE audit_items AS target SET last_execution_item_id = NULL
FROM batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_review_decisions WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_review_decisions AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_review_requests WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_review_requests AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_finding_assessments WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_finding_assessments AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_finding_contributions WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_finding_contributions AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_findings WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_findings AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_execution_items WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_execution_items AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_items WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_items AS target USING batch WHERE target.ctid = batch.ctid`,
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
