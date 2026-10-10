package runstore

// SQL statements for delete_store.go.

// lockRunForDeletionSQL locks Run $1 owned by $2 FOR UPDATE and returns its
// state, publication mode and Audit execution authority.
// Used by PostgresStore.LockRunForDeletion.
var lockRunForDeletionSQL = `
SELECT state, publication_mode, audit_execution_id
FROM workflow_runs
WHERE run_id = $1 AND owner_id = $2
FOR UPDATE`

// allocationReleasePendingExpr is true while any Stage allocation of the Run
// row aliased run has not completed its release.
const allocationReleasePendingExpr = `
EXISTS (
    SELECT 1
      FROM stage_executions AS execution
      JOIN stage_allocations AS allocation
        ON allocation.stage_execution_id = execution.stage_execution_id
     WHERE execution.run_id = run.run_id
       AND allocation.release_completed_at IS NULL
)`

// runAllocationReleasePendingSQL reports allocationReleasePendingExpr for
// Run $1. Used by PostgresStore.LockRunForDeletion.
var runAllocationReleasePendingSQL = `
SELECT ` + allocationReleasePendingExpr + `
  FROM workflow_runs AS run
 WHERE run.run_id = $1`

// readRunDeletionGatesSQL reads, without locking, the Run-owned deletion
// gates of owner $1's Runs among IDs $2: state, publication mode, Audit
// execution authority and allocationReleasePendingExpr.
// Used by PostgresStore.ReadRunDeletionGates.
var readRunDeletionGatesSQL = `
SELECT run.run_id, run.state, run.publication_mode, run.audit_execution_id,
       ` + allocationReleasePendingExpr + `
  FROM workflow_runs AS run
 WHERE run.owner_id = $1 AND run.run_id = ANY($2::text[])`
