package evalstore

// SQL statements for records.go.

// latestNativeRecordSQL returns the digest of the newest eval_records row of
// kind $4 for a member of the owner's experiment that the native collector
// (actor 'system:eval-collector') wrote, with the digest as tie-breaker.
// Used by Store.LatestNativeRecord.
var latestNativeRecordSQL = `
SELECT r.record_sha256
FROM eval_records r
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id = $1
    AND e.experiment_id = $2
    AND r.member_id = $3
    AND r.kind = $4
    AND r.actor_id = 'system:eval-collector'
ORDER BY r.created_at DESC,r.record_sha256 DESC
LIMIT 1
`
