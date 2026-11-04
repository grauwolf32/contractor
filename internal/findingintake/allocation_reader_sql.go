package findingintake

// allocationFindingContextSQL derives the active Project and pinned engagement
// through the exact durable allocation; no request field can widen authority.
const allocationFindingContextSQL = `SELECT run.owner_id,run.project_id,
 CASE WHEN audit.baseline_snapshot->'engagement' IS NULL OR audit.baseline_snapshot->'engagement'='null'::jsonb THEN '' ELSE coalesce(audit.baseline_snapshot #>> '{engagement,document,accessMode}','invalid') END
 FROM stage_allocations AS allocation
 JOIN stage_executions AS stage USING(stage_execution_id)
 JOIN workflow_runs AS run ON run.run_id=stage.run_id
 JOIN projects AS project ON project.project_id=run.project_id AND project.owner_id=run.owner_id
 LEFT JOIN audit_executions AS execution ON execution.execution_id=run.audit_execution_id AND execution.run_id=run.run_id
 LEFT JOIN audits AS audit ON audit.audit_id=execution.audit_id AND audit.owner_id=run.owner_id AND audit.project_id=run.project_id
 WHERE allocation.allocation_id=$1 AND stage.stage_execution_id=$2 AND run.run_id=$3
 AND allocation.logical_agent_name=$4 AND allocation.namespace=$5
 AND allocation.runtime_agent_id=$6 AND allocation.runtime_agent_instance_id=$7
 AND allocation.release_completed_at IS NULL AND project.kind='project' AND project.lifecycle_state='active'
 AND (run.audit_execution_id IS NULL OR audit.state IN ('active','paused'))`

// Automatic blackbox discovery only discloses a source whose owning retained
// Audit explicitly pins blackbox provenance. Deleted/unknown origins fail closed.
const blackboxFindingsSQL = `SELECT coalesce(bool_and(coalesce(audit.baseline_snapshot #>> '{engagement,document,accessMode}'='blackbox',false)),true)
 FROM project_findings AS finding
 JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id=finding.first_receipt_id
 LEFT JOIN audits AS audit ON audit.audit_id=receipt.audit_id AND audit.owner_id=receipt.owner_id AND audit.project_id=receipt.project_id
 WHERE finding.project_id=$2 AND receipt.owner_id=$1 AND finding.finding_id=ANY($3::text[])`

// Check the entire selected receipt set before any document is materialized.
const blackboxReceiptsSQL = `SELECT coalesce(bool_and(coalesce(audit.baseline_snapshot #>> '{engagement,document,accessMode}'='blackbox',false)),true)
 FROM finding_proposal_receipts AS receipt
 LEFT JOIN audits AS audit ON audit.audit_id=receipt.audit_id AND audit.owner_id=receipt.owner_id AND audit.project_id=receipt.project_id
 WHERE receipt.owner_id=$1 AND receipt.receipt_id=ANY($2::text[])`
