ALTER TABLE workflow_runs
    ADD COLUMN skill_snapshot jsonb NOT NULL DEFAULT '[]'::jsonb,
    ADD CONSTRAINT workflow_runs_skill_snapshot_array
        CHECK (jsonb_typeof(skill_snapshot) = 'array');

CREATE INDEX workflow_runs_skill_initialization_scheduler_idx
    ON workflow_runs (scheduler_claim_expires_at, created_at, run_id)
    WHERE state = 'initializing'
      AND state_reason_code = 'skill_initialization_pending';

CREATE OR REPLACE FUNCTION contractor_protect_workflow_run_skill_snapshot()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.skill_snapshot IS DISTINCT FROM OLD.skill_snapshot
        AND (OLD.state <> 'initializing' OR NEW.state <> 'initializing')
    THEN
        RAISE EXCEPTION 'terminal WorkflowRun Skill snapshot cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER workflow_runs_protect_skill_snapshot
BEFORE UPDATE ON workflow_runs
FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_skill_snapshot();
