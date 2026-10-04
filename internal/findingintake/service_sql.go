package findingintake

// SQL statements for service.go.

// verifyTrustedExecutionSQL reports whether allocation $1 is still unreleased
// and belongs to stage execution $2 of Run $3 with logical agent $4, namespace
// $5, runtime agent $6 and runtime instance $7, i.e. whether the grant matches
// live control-plane state. Used by verifyTrustedExecution.
var verifyTrustedExecutionSQL = `
SELECT EXISTS (
    SELECT 1
      FROM stage_allocations AS allocation
      JOIN stage_executions AS execution
        ON execution.stage_execution_id = allocation.stage_execution_id
     WHERE allocation.allocation_id = $1
       AND allocation.stage_execution_id = $2
       AND execution.run_id = $3
       AND allocation.logical_agent_name = $4
       AND allocation.namespace = $5
       AND allocation.runtime_agent_id = $6
       AND allocation.runtime_agent_instance_id = $7
       AND allocation.release_completed_at IS NULL
)`

// insertReceiptSQL inserts one finding_proposal_receipts row recording the
// proposal and its evidence, the submitting allocation and agent, the Run and
// Project, the optional Audit execution origin and the Workflow identity. The
// caller maps a unique violation to a conflict. Used by insertReceipt.
var insertReceiptSQL = `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id, runtime_instance_id,
    stage_execution_id, logical_agent_name, invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id, audit_execution_id, audit_id, audit_role,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type, proposal_size_bytes, evidence
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22,
    $23, $24, $25, $26, $27
)`
