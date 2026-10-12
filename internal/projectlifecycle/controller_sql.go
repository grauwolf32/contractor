package projectlifecycle

// SQL statements for controller.go.

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
