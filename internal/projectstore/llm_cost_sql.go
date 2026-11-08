package projectstore

const lockProjectCostSQL = `SELECT llm_cost FROM projects WHERE owner_id=$1 AND project_id=$2 FOR UPDATE`
const updateProjectCostSQL = `UPDATE projects SET llm_cost=$2::jsonb WHERE project_id=$1`

// readProjectCostSQL uses prepared totals; the existing membership indexes
// answer only existence probes, never a per-Run cost aggregation.
const readProjectCostSQL = `
SELECT project.llm_cost,
       EXISTS(SELECT 1 FROM workflow_runs run WHERE run.project_id=project.project_id
              AND run.state NOT IN ('succeeded','failed','cancelled'))
       OR EXISTS(SELECT 1 FROM audits audit WHERE audit.project_id=project.project_id
              AND audit.state IN ('active','finalizing','cancelling')),
       EXISTS(SELECT 1 FROM workflow_runs run WHERE run.project_id=project.project_id)
FROM projects project WHERE project.owner_id=$1 AND project.project_id=$2`
