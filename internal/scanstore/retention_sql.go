package scanstore

// expiredSweepSQL selects one terminal resource and joins any controller work
// by locking its parent. In-progress physical cleanup takes priority.
const expiredSweepSQL = `SELECT owner_id,sweep_id FROM scan_sweeps
WHERE (purging OR settled_at <= $1) AND state IN ('completed','cancelled')
AND (holder_id IS NULL OR expires_at <= clock_timestamp())
ORDER BY purging DESC,settled_at,sweep_id LIMIT 1 FOR UPDATE SKIP LOCKED`

// purgeMemberBatchSQL never deletes child Runs or their artifacts. At most
// 25 rows leave in each maintenance transaction.
const purgeMemberBatchSQL = `WITH batch AS (
 SELECT ordinal FROM scan_sweep_members WHERE sweep_id=$1 ORDER BY ordinal LIMIT $2
)
DELETE FROM scan_sweep_members m USING batch
WHERE m.sweep_id=$1 AND m.ordinal=batch.ordinal`
