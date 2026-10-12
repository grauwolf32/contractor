package evalstore

// SQL statements for claims.go.

// claimStatement leases up to $3 experiments with actionable work (an active
// state, a paused deadline that has passed, an accepted or running command, or
// an unpublished projection revision) to holder $1 for $2 milliseconds. Only
// free or expired eval_controller_claims rows qualify; they are locked FOR
// UPDATE SKIP LOCKED and leased through the shared claim helper, which
// advances their epoch as the fencing token and returns the new claims.
// Parameterized lookups avoid scanning retained terminal populations. A
// materialized ordered stream feeds the locking lookup, so LIMIT stops
// locking after the requested batch (including SKIP LOCKED rows).
// Used by Store.Claim.
var claimStatement = `
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
), ordered AS MATERIALIZED (
    SELECT a.experiment_id
    FROM actionable a
    JOIN LATERAL (
        SELECT claim.epoch, experiment.updated_at
        FROM eval_controller_claims claim
        JOIN eval_experiments experiment USING (experiment_id)
        WHERE claim.experiment_id = a.experiment_id
          AND ` + evalClaims.AvailableCondition("claim") + `
        OFFSET 0
    ) eligible ON true
    ORDER BY eligible.epoch, eligible.updated_at, a.experiment_id
), candidates AS (
    SELECT locked.claim_tid
    FROM ordered
    JOIN LATERAL (
        SELECT claim.ctid AS claim_tid
        FROM eval_controller_claims claim
        WHERE claim.experiment_id = ordered.experiment_id
          AND ` + evalClaims.AvailableCondition("claim") + `
        FOR UPDATE SKIP LOCKED
    ) locked ON true
    LIMIT $3
)` + evalClaims.LeaseSQL()
