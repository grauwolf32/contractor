package pgstore

// At most 128 primary records change in a slice. The caller returns after the
// first nonempty step, so clearing a graph never depends on deletion order.
var projectFindingPurgeSQL = []string{
	`WITH batch AS (SELECT finding_id FROM project_findings WHERE project_id=$1 AND duplicate_target_id IS NOT NULL ORDER BY finding_id LIMIT 128)
 UPDATE project_findings AS finding SET state=NULL,analyst_verdict=NULL,analyst_severity=NULL,rationale=NULL,
 actor_id=NULL,assessed_at=NULL,duplicate_target_id=NULL,revision=revision+1,updated_at=clock_timestamp()
 FROM batch WHERE finding.finding_id=batch.finding_id`,
	`WITH batch AS (SELECT receipt_id FROM project_finding_sources WHERE project_id=$1 ORDER BY receipt_id LIMIT 128)
 DELETE FROM project_finding_sources AS source USING batch WHERE source.receipt_id=batch.receipt_id`,
	`WITH batch AS (SELECT finding_id FROM project_findings WHERE project_id=$1 ORDER BY finding_id LIMIT 128)
 DELETE FROM project_findings AS finding USING batch WHERE finding.finding_id=batch.finding_id`,
}
