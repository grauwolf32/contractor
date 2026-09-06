-- Manual continuation preserves terminal attempts and the original fail decision.
CREATE TABLE run_stage_resumptions (
    source_execution_id text PRIMARY KEY REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE,
    run_id text NOT NULL REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
    target_execution_id text NOT NULL UNIQUE REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE,
    requested_by text NOT NULL,
    requested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    transaction_id xid8 NOT NULL DEFAULT pg_current_xact_id()
);

CREATE FUNCTION contractor_protect_run_stage_resumption()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Run continuation receipts are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER run_stage_resumptions_immutable
BEFORE UPDATE ON run_stage_resumptions
FOR EACH ROW EXECUTE FUNCTION contractor_protect_run_stage_resumption();

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
                  AND run.state = 'running' AND run.publication_mode = 'ordinary'
                  AND target.run_id = run.run_id AND target.state = 'preparing'
            )
        ) THEN
            RAISE EXCEPTION 'frozen Artifact binding cannot be changed' USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
