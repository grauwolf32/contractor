-- Collection policy is immutable allocation provenance. NULL is retained only
-- for allocations created before this migration and projects as "legacy".
ALTER TABLE stage_allocations
    ADD COLUMN performance_collection_policy text,
    ADD CONSTRAINT stage_allocation_performance_collection_policy CHECK (
        performance_collection_policy IS NULL OR
        performance_collection_policy IN ('requested', 'disabled', 'unsupported')
    );

CREATE INDEX stage_executions_terminal_history_idx
    ON stage_executions (terminal_at DESC, stage_execution_id DESC)
    WHERE terminal_at IS NOT NULL;

CREATE INDEX stage_executions_run_terminal_history_idx
    ON stage_executions (run_id, terminal_at DESC, stage_execution_id DESC)
    WHERE terminal_at IS NOT NULL;

CREATE OR REPLACE FUNCTION contractor_protect_stage_allocation_performance_policy()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.performance_collection_policy IS DISTINCT FROM OLD.performance_collection_policy THEN
        RAISE EXCEPTION 'Stage allocation performance collection policy is immutable'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER stage_allocations_protect_performance_policy
BEFORE UPDATE ON stage_allocations
FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_allocation_performance_policy();
