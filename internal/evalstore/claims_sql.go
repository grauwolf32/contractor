package evalstore

// SQL statements for claims.go.

// claimStatement leases up to $3 experiments with actionable work (an active
// state, a paused deadline that has passed, an accepted or running command, or
// an unpublished projection revision) to holder $1 for $2 milliseconds. Only
// free or expired eval_controller_claims rows qualify; they are locked FOR
// UPDATE SKIP LOCKED and their epoch is bumped as the fencing token. Returns
// the new claims. Used by Store.Claim.
const claimStatement = `
WITH actionable AS (
    SELECT experiment_id FROM eval_experiments
    WHERE state IN ('preparing', 'running', 'settling', 'pausing', 'cancelling')
    UNION
    SELECT experiment_id FROM eval_experiments
    WHERE state = 'paused' AND deadline_at <= statement_timestamp()
    UNION
    SELECT experiment_id FROM eval_commands
    WHERE state IN ('accepted', 'running')
    UNION
    SELECT experiment_id FROM eval_projection_queue
    WHERE revision <> published_revision
), candidates AS (
    SELECT c.experiment_id
    FROM actionable a
    JOIN eval_controller_claims c USING (experiment_id)
    JOIN eval_experiments e USING (experiment_id)
    WHERE c.holder_id IS NULL OR c.expires_at <= clock_timestamp()
    ORDER BY c.epoch, e.updated_at, e.experiment_id
    FOR UPDATE OF c SKIP LOCKED
    LIMIT $3
)
UPDATE eval_controller_claims c
SET epoch = epoch + 1, holder_id = $1,
    expires_at = clock_timestamp() + $2::bigint * interval '1 millisecond'
FROM candidates x
WHERE c.experiment_id = x.experiment_id
RETURNING c.experiment_id, c.holder_id, c.epoch, c.expires_at
`
