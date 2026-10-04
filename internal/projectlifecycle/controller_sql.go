package projectlifecycle

// SQL statements for controller.go.

// claimProjectDeletionSQL leases the oldest 'deleting' project whose deletion
// claim is absent or expired, using FOR UPDATE SKIP LOCKED so controllers do
// not contend. It stores claim id $1 with an expiry of $2 microseconds, or $3
// during the purge phases $4/$5, and returns project_id, owner_id and
// deletion_phase. Used by Controller.claim.
var claimProjectDeletionSQL = `
WITH candidate AS (
    SELECT project_id
    FROM projects
    WHERE lifecycle_state = 'deleting'
      AND (deletion_claim_id IS NULL OR deletion_claim_expires_at <= clock_timestamp())
    ORDER BY deletion_requested_at, project_id
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE projects AS project
SET deletion_claim_id = $1,
    deletion_claimed_at = clock_timestamp(),
    deletion_claim_expires_at = clock_timestamp() + (CASE
        WHEN project.deletion_phase IN ($4, $5) THEN $3::bigint ELSE $2::bigint
    END * interval '1 microsecond')
FROM candidate
WHERE project.project_id = candidate.project_id
RETURNING project.project_id, project.owner_id, project.deletion_phase`

// waitForDrainSQL reports whether project $1 still blocks purging: it has a
// non-terminal workflow run, a stage allocation of one of its runs whose
// release has not completed, or any audit. Returns a single boolean.
// Used by Controller.waitForDrain.
var waitForDrainSQL = `
SELECT EXISTS (
    SELECT 1 FROM workflow_runs
    WHERE project_id = $1
      AND state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling')
) OR EXISTS (
    SELECT 1
    FROM workflow_runs AS run
    JOIN stage_executions AS execution ON execution.run_id = run.run_id
    JOIN stage_allocations AS allocation
      ON allocation.stage_execution_id = execution.stage_execution_id
    WHERE run.project_id = $1
      AND allocation.release_completed_at IS NULL
) OR EXISTS (
    SELECT 1 FROM audits WHERE project_id = $1
)`
