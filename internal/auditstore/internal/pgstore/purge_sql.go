package pgstore

// purgeAuditDomainSQL clears derived back-references before deleting their
// immutable targets, then removes leaf rows before parents. The caller shares
// one remaining-row budget across all statements and commits each pass. Empty
// tables use their Audit FK indexes; pointer resets use partial purge indexes.
var purgeAuditDomainSQL = []string{
	`WITH batch AS (SELECT audit_id,finding_id FROM project_finding_audits WHERE audit_id=$1 ORDER BY finding_id LIMIT LEAST($2,128))
 DELETE FROM project_finding_audits AS membership USING batch WHERE membership.audit_id=batch.audit_id AND membership.finding_id=batch.finding_id`,

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
	`WITH batch AS (
  SELECT ctid FROM audit_executions WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_executions AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_artifact_links WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_artifact_links AS target USING batch WHERE target.ctid = batch.ctid`,
	`WITH batch AS (
  SELECT ctid FROM audit_events WHERE audit_id = $1 LIMIT $2
)
DELETE FROM audit_events AS target USING batch WHERE target.ctid = batch.ctid`,
}
