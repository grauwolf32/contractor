package runstore

// costRunContextSQL locks only the existing Run; all actor and parent updates
// share the composition's transaction. Project and Audit links are immutable.
const costRunContextSQL = `
SELECT run.run_id,run.owner_id,run.project_id,run.audit_execution_id,run.llm_cost
FROM stage_executions stage JOIN workflow_runs run USING(run_id)
WHERE stage.stage_execution_id=$1 FOR UPDATE OF run`

const costPlannerContributionSQL = `SELECT planner_llm_cost FROM stage_executions WHERE stage_execution_id=$1 FOR UPDATE`
const costAllocationContributionSQL = `SELECT llm_cost FROM stage_allocations WHERE stage_execution_id=$1 AND allocation_id=$2 FOR UPDATE`
const updatePlannerCostContributionSQL = `UPDATE stage_executions SET planner_llm_cost=$2::jsonb WHERE stage_execution_id=$1`
const updateAllocationCostContributionSQL = `UPDATE stage_allocations SET llm_cost=$3::jsonb WHERE stage_execution_id=$1 AND allocation_id=$2`
const updateRunCostSQL = `UPDATE workflow_runs SET llm_cost=$2::jsonb WHERE run_id=$1`

// readRunCostSQL reads one bounded parent field, independently of Run ETags.
const readRunCostSQL = `
SELECT run.llm_cost,run.state NOT IN ('succeeded','failed','cancelled'),
       EXISTS(SELECT 1 FROM stage_executions stage WHERE stage.run_id=run.run_id)
FROM workflow_runs run WHERE run.owner_id=$1 AND run.run_id=$2`
