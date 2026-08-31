ALTER TABLE stage_allocations
    ADD COLUMN release_attempted_at timestamptz,
    ADD COLUMN release_completed_at timestamptz,
    ADD CONSTRAINT stage_allocation_release_timestamps CHECK (
        release_completed_at IS NULL
        OR release_attempted_at IS NOT NULL AND release_completed_at >= release_attempted_at
    );

CREATE INDEX stage_allocations_pending_release_idx
    ON stage_allocations (release_attempted_at, stage_execution_id, allocation_id)
    WHERE release_completed_at IS NULL;
