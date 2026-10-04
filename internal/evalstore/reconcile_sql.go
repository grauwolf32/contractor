package evalstore

// SQL statements for reconcile.go.

// reconciliationCandidatesSQL returns up to $3 'intent' and up to $3 'accepted'
// submission member IDs of the owner's experiment, each lane least recently
// updated first (member_id breaks ties), with all intents ahead of accepted
// members. Used by Store.ReconciliationCandidates.
var reconciliationCandidatesSQL = `
WITH intents AS (
    SELECT s.member_id, s.updated_at
    FROM eval_submissions s
    JOIN eval_experiments e USING(experiment_id)
    WHERE e.owner_id=$1 AND e.experiment_id=$2 AND s.state='intent'
    ORDER BY s.updated_at, s.member_id LIMIT $3
), accepted AS (
    SELECT s.member_id, s.updated_at
    FROM eval_submissions s
    JOIN eval_experiments e USING(experiment_id)
    WHERE e.owner_id=$1 AND e.experiment_id=$2 AND s.state='accepted'
    ORDER BY s.updated_at, s.member_id LIMIT $3
)
SELECT member_id FROM (
    SELECT member_id, updated_at, 0 AS phase FROM intents
    UNION ALL
    SELECT member_id, updated_at, 1 AS phase FROM accepted
) work
ORDER BY phase, updated_at, member_id
`
