package pgstore

// Each slice changes at most 128 primary records and 1 MiB of metadata.
// Six times text byte length bounds JSON escaping; the fixed allowance covers
// bounded IDs, actor, canonical fields and timestamps. Even the largest row
// fits alone. Candidate work is also bounded before summing sizes.
func projectFindingPurgeBatchSQL(duplicates bool) string {
	predicate, order := "", "finding_id"
	if duplicates {
		predicate = " AND duplicate_target_id IS NOT NULL"
		order = "duplicate_target_id"
	}
	return `WITH candidates AS MATERIALIZED (
 SELECT finding_id,8192+6*(octet_length(title)+coalesce(octet_length(rationale),0))
 +octet_length(first_proposal_ref::text) AS bytes FROM project_findings
 WHERE project_id=$1` + predicate + ` ORDER BY ` + order + ` LIMIT 128
 ), batch AS (
 SELECT finding_id FROM (SELECT finding_id,sum(bytes) OVER(ORDER BY finding_id ROWS UNBOUNDED PRECEDING) AS bytes FROM candidates) AS sized
 WHERE bytes<=1048576
 ) `
}

// Return after the first nonempty step; clearing a duplicate graph precedes
// source/Finding removal and never depends on the target's deletion order.
var projectFindingPurgeSQL = []string{
	projectFindingPurgeBatchSQL(true) + `UPDATE project_findings AS finding
 SET state=NULL,analyst_verdict=NULL,analyst_severity=NULL,rationale=NULL,
 actor_id=NULL,assessed_at=NULL,duplicate_target_id=NULL,revision=revision+1,updated_at=clock_timestamp()
 FROM batch WHERE finding.finding_id=batch.finding_id`,
	`WITH batch AS (SELECT receipt_id FROM project_finding_sources WHERE project_id=$1 ORDER BY receipt_id LIMIT 128)
 DELETE FROM project_finding_sources AS source USING batch WHERE source.receipt_id=batch.receipt_id`,
	projectFindingPurgeBatchSQL(false) + `DELETE FROM project_findings AS finding USING batch WHERE finding.finding_id=batch.finding_id`,
}
