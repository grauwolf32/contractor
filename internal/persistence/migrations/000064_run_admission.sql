ALTER TABLE workflow_runs DROP CONSTRAINT workflow_runs_state_check;
ALTER TABLE workflow_runs ADD CONSTRAINT workflow_runs_state_check CHECK (
    state IN ('initializing','pending','running','waiting','cancelling','succeeded','failed','cancelled')
);
ALTER TABLE stage_executions ADD COLUMN admitted_at timestamptz;
-- Historical execution timestamps remain intact. Queued work has not started.
UPDATE stage_executions SET admitted_at = COALESCE(planner_started_at, created_at)
WHERE state <> 'preparing' OR EXISTS (
    SELECT 1 FROM stage_allocations a WHERE a.stage_execution_id = stage_executions.stage_execution_id
);
UPDATE workflow_runs r SET state='pending', started_at=NULL, state_reason_code='awaiting_admission'
WHERE r.state='running' AND NOT EXISTS (
    SELECT 1 FROM stage_executions s WHERE s.run_id=r.run_id AND s.admitted_at IS NOT NULL
);
CREATE INDEX workflow_runs_admission_idx ON workflow_runs (state, scheduler_claim_expires_at, updated_at, created_at, run_id)
WHERE state IN ('pending','running','waiting','cancelling');

CREATE OR REPLACE FUNCTION contractor_protect_artifact_binding()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.scope_kind IS DISTINCT FROM OLD.scope_kind
        OR NEW.scope_id IS DISTINCT FROM OLD.scope_id
        OR NEW.namespace IS DISTINCT FROM OLD.namespace
        OR NEW.name IS DISTINCT FROM OLD.name
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'Artifact binding identity cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.frozen AND (
        NEW.current_revision IS DISTINCT FROM OLD.current_revision OR NOT NEW.frozen
    ) THEN
        -- Only the atomic continuation transaction may thaw an output binding;
        -- it must retain the exact current revision. Other frozen bindings and
        -- all immutable revisions retain their original protection.
        IF NOT (
            OLD.scope_kind = 'run' AND OLD.namespace = 'outputs'
            AND NOT NEW.frozen AND NEW.current_revision = OLD.current_revision
            AND EXISTS (
                SELECT 1 FROM run_stage_resumptions AS receipt
                JOIN workflow_runs AS run ON run.run_id = receipt.run_id
                JOIN stage_executions AS target ON target.stage_execution_id = receipt.target_execution_id
                WHERE receipt.run_id = OLD.scope_id
                  AND receipt.transaction_id = pg_current_xact_id()
                  AND run.state = 'pending' AND run.publication_mode = 'ordinary'
                  AND target.run_id = run.run_id AND target.state = 'preparing'
            )
        ) THEN
            RAISE EXCEPTION 'frozen Artifact binding cannot be changed' USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
