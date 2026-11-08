package pgstore

const lockAuditCostSQL = `
SELECT audit.audit_id,audit.llm_cost FROM audit_executions execution
JOIN audits audit USING(audit_id)
WHERE execution.execution_id=$1 AND execution.run_id=$2
FOR UPDATE OF audit,execution`
const updateAuditCostSQL = `UPDATE audits SET llm_cost=$2::jsonb WHERE audit_id=$1`
const updateAuditExecutionCostSQL = `UPDATE audit_executions SET llm_cost=$2::jsonb WHERE execution_id=$1`

// readAuditCostSQL has independent operational freshness; cost never changes
// the authoritative Audit DTO/revision. The execution existence probe is indexed.
const readAuditCostSQL = `
SELECT audit.llm_cost,
       audit.state IN ('active','finalizing','cancelling')
       OR EXISTS(SELECT 1 FROM audit_executions execution JOIN workflow_runs run USING(run_id)
                 WHERE execution.audit_id=audit.audit_id AND run.state NOT IN ('succeeded','failed','cancelled')),
       EXISTS(SELECT 1 FROM audit_executions execution WHERE execution.audit_id=audit.audit_id)
FROM audits audit WHERE audit.owner_id=$1 AND audit.audit_id=$2`

// Only explicitly requested attempts are read (at most 64); this does not scan
// an Audit history or change its authoritative revision.
const readAuditExecutionCostsSQL = `
SELECT execution.execution_id,run.run_id,execution.llm_cost,
       COALESCE(run.state NOT IN ('succeeded','failed','cancelled'),false)
FROM audits audit JOIN audit_executions execution USING(audit_id)
LEFT JOIN workflow_runs run USING(run_id)
WHERE audit.owner_id=$1 AND audit.audit_id=$2 AND execution.execution_id=ANY($3::text[])
ORDER BY execution.execution_id`
