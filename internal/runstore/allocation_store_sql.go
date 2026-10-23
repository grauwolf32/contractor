package runstore

// recordStageAllocationSQL pins identity, configuration and actual location.
// Used by PostgresStore.RecordStageAllocation; conflict reconciliation compares every immutable field.
const recordStageAllocationSQL = `
INSERT INTO stage_allocations (
    allocation_id, stage_execution_id, logical_agent_name, namespace,
    agent_template_ref, worker_runtime_ref,
    runtime_agent_id, runtime_agent_instance_id, runtime_agent_label_revision,
    runtime_configuration_schema_version, runtime_configuration,
    performance_collection_policy, completion_contract, runtime_placement_labels
) VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9, $10, $11::jsonb, $12, $13::jsonb, $14)
ON CONFLICT DO NOTHING
RETURNING allocation_id`

// listStageAllocationsSQL reads one bounded Stage set with retained location declarations.
// Used by PostgresStore.listStageAllocations.
const listStageAllocationsSQL = `
SELECT allocation_id, stage_execution_id, logical_agent_name, namespace,
       agent_template_ref, worker_runtime_ref,
       runtime_agent_id, runtime_agent_instance_id, runtime_agent_label_revision::text,
       runtime_configuration_schema_version, runtime_configuration,
       performance_collection_policy, completion_contract, created_at,
       release_attempted_at, release_completed_at, runtime_placement_labels
FROM stage_allocations
WHERE stage_execution_id = ANY($1::text[])
ORDER BY stage_execution_id, logical_agent_name`
