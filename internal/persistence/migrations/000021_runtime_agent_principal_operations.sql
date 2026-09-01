ALTER TABLE runtime_management_operations
    DROP CONSTRAINT runtime_management_operations_resource_id_check;

ALTER TABLE runtime_management_operations
    ADD CONSTRAINT runtime_management_operations_resource_id_check CHECK (
        resource_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
    );

CREATE INDEX stage_allocations_live_runtime_agent_idx
    ON stage_allocations (runtime_agent_id, allocation_id)
    WHERE runtime_agent_id IS NOT NULL AND release_completed_at IS NULL;
