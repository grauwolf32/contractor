package projectlifecycle

// SQL statements for controller.go.

// waitForDrainSQL reports whether project $1 of owner $2 still blocks
// purging: it has a non-terminal workflow run, a stage allocation of one of
// its runs whose release has not completed, or any audit. Returns a single
// boolean. Used by Controller.waitForDrain on every poll of a draining
// Project.
//
// Each check visits the Project's own rows or the unreleased allocations, not
// the Run history or every owner's active Runs (V407-025). The owner drops no
// Run: workflow_runs_project_owner_fkey ties every Run of a Project to the
// Project's one owner.
//   - the active Run check names the owner as well as the Project, so the
//     planner finds the Project's Runs through
//     workflow_runs_owner_project_created_idx, which the Project alone cannot
//     use because the owner leads it, and visits no Run of another Project.
//     On the test's fixture the planner intersects those index entries with
//     the entries of the active Runs (workflow_runs_runtime_credentials_idx),
//     so the check reads one index entry for each Run of the Project and for
//     each active Run, but no Run row unless one of the Project's Runs is
//     active;
//   - the allocation check starts from the allocations whose release has not
//     completed (stage_allocations_pending_release_idx): the live ones and
//     those whose release is being retried. It reads one Run per allocation
//     and keeps the Project alone: the owner adds no selectivity, but the
//     planner treats the two as independent, underestimates the Project's
//     Runs and would start from the Project's whole Stage history instead;
//   - the Audit check reads audits_project_state_idx.
//
// TestProjectDrainChecksReadOnlyTheProjectAndUnreleasedAllocations holds
// these plans on a realistic fixture.
var waitForDrainSQL = `
SELECT EXISTS (
    SELECT 1 FROM workflow_runs
    WHERE owner_id = $2 AND project_id = $1
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

// purgeRemainsSQL reports whether project $1 of owner $2 still has any
// workflow run and whether it still has any audit. Every Project purge
// transaction rechecks both, and by then the Project has none, so each
// check must find nothing through an index led by what it names: the owner
// and the Project for Runs (workflow_runs_owner_project_created_idx; with the
// Project alone the planner walks a whole index of the Run history), the
// Project for Audits (audits_project_state_idx). As in waitForDrainSQL, the
// owner drops no Run. Used by Controller.purgeProject.
var purgeRemainsSQL = `
SELECT EXISTS (SELECT 1 FROM workflow_runs WHERE owner_id = $2 AND project_id = $1),
       EXISTS (SELECT 1 FROM audits WHERE project_id = $1)`
