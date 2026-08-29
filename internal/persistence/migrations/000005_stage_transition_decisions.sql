ALTER TABLE stage_executions
    ADD CONSTRAINT stage_execution_run_identity UNIQUE (run_id, stage_execution_id);

CREATE TABLE stage_transition_decisions (
    source_execution_id text PRIMARY KEY,
    run_id text NOT NULL,
    action text NOT NULL CHECK (action IN ('next', 'retry', 'succeed', 'fail')),
    target_stage_name text,
    target_execution_id text,
    decided_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT stage_transition_source_run
        FOREIGN KEY (run_id, source_execution_id)
        REFERENCES stage_executions(run_id, stage_execution_id)
        ON DELETE CASCADE,
    CONSTRAINT stage_transition_target_run_stage
        FOREIGN KEY (run_id, target_stage_name, target_execution_id)
        REFERENCES stage_executions(run_id, stage_name, stage_execution_id)
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT stage_transition_target_once UNIQUE (target_execution_id),
    CONSTRAINT stage_transition_decision_shape CHECK (
        (
            action IN ('next', 'retry')
            AND target_stage_name IS NOT NULL AND btrim(target_stage_name) <> ''
            AND target_execution_id IS NOT NULL AND btrim(target_execution_id) <> ''
        ) OR (
            action IN ('succeed', 'fail')
            AND target_stage_name IS NULL
            AND target_execution_id IS NULL
        )
    )
);

CREATE INDEX stage_transition_decisions_run_idx
    ON stage_transition_decisions (run_id, decided_at, source_execution_id);

CREATE OR REPLACE FUNCTION contractor_protect_stage_transition_decision_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'Stage transition decisions are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER stage_transition_decisions_protect_immutable
BEFORE UPDATE ON stage_transition_decisions
FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_transition_decision_immutable();
