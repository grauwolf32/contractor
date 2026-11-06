-- Keep one immutable automatic decision on its source Stage and one manual
-- continuation receipt on its target. Preserve historical timestamps and xid8
-- authority exactly; output thaw still requires the creating transaction.
ALTER TABLE stage_executions
    ADD COLUMN transition_action text,
    ADD COLUMN transition_target_stage_name text,
    ADD COLUMN transition_target_execution_id text,
    ADD COLUMN transition_escalation_ordinal integer,
    ADD COLUMN transition_escalation_exhausted boolean NOT NULL DEFAULT false,
    ADD COLUMN transition_decided_at timestamptz,
    ADD COLUMN resume_requested_by text,
    ADD COLUMN resume_requested_at timestamptz,
    ADD COLUMN resume_transaction_id xid8;

UPDATE stage_executions stage
SET transition_action = decision.action,
    transition_target_stage_name = decision.target_stage_name,
    transition_target_execution_id = decision.target_execution_id,
    transition_escalation_ordinal = decision.escalation_ordinal,
    transition_escalation_exhausted = decision.escalation_exhausted,
    transition_decided_at = decision.decided_at
FROM stage_transition_decisions decision
WHERE stage.stage_execution_id = decision.source_execution_id;

UPDATE stage_executions stage
SET resume_requested_by = receipt.requested_by,
    resume_requested_at = receipt.requested_at,
    resume_transaction_id = receipt.transaction_id
FROM run_stage_resumptions receipt
WHERE stage.stage_execution_id = receipt.target_execution_id;

ALTER TABLE stage_executions DROP CONSTRAINT stage_execution_resume_receipt;
DROP TABLE stage_transition_decisions;
DROP TABLE run_stage_resumptions;
DROP FUNCTION contractor_protect_stage_transition_decision_immutable();
DROP FUNCTION contractor_protect_run_stage_resumption();
DROP FUNCTION contractor_validate_run_stage_resumption();

ALTER TABLE stage_executions
    ADD CONSTRAINT stage_execution_transition_presence CHECK (
        (transition_action IS NULL AND transition_target_stage_name IS NULL
         AND transition_target_execution_id IS NULL AND transition_decided_at IS NULL
         AND transition_escalation_ordinal IS NULL AND NOT transition_escalation_exhausted)
        OR (transition_action IS NOT NULL AND transition_decided_at IS NOT NULL)),
    ADD CONSTRAINT stage_transition_action CHECK ((transition_action = ANY (ARRAY['next'::text, 'retry'::text, 'escalate'::text, 'succeed'::text, 'fail'::text]))),
    ADD CONSTRAINT stage_transition_decision_shape CHECK ((((transition_action = ANY (ARRAY['next'::text, 'retry'::text, 'escalate'::text])) AND (transition_target_stage_name IS NOT NULL) AND (btrim(transition_target_stage_name) <> ''::text) AND (transition_target_execution_id IS NOT NULL) AND (btrim(transition_target_execution_id) <> ''::text)) OR ((transition_action = ANY (ARRAY['succeed'::text, 'fail'::text])) AND (transition_target_stage_name IS NULL) AND (transition_target_execution_id IS NULL)))),
    ADD CONSTRAINT stage_transition_escalation_shape CHECK ((((transition_action = 'escalate'::text) AND (transition_escalation_ordinal IS NOT NULL) AND (transition_escalation_ordinal > 0) AND (transition_escalation_exhausted = false)) OR ((transition_action = ANY (ARRAY['next'::text, 'fail'::text])) AND (transition_escalation_ordinal IS NOT NULL) AND (transition_escalation_ordinal > 0) AND (transition_escalation_exhausted = true)) OR ((transition_action <> 'escalate'::text) AND (transition_escalation_ordinal IS NULL) AND (transition_escalation_exhausted = false)))),
    ADD CONSTRAINT stage_transition_target_run_stage
        FOREIGN KEY (run_id, transition_target_stage_name, transition_target_execution_id)
        REFERENCES stage_executions(run_id, stage_name, stage_execution_id)
        DEFERRABLE INITIALLY DEFERRED,
    ADD CONSTRAINT stage_execution_transition_target_once UNIQUE (transition_target_execution_id),
    ADD CONSTRAINT stage_execution_resume_source_once UNIQUE (resume_source_execution_id),
    ADD CONSTRAINT stage_execution_resume_receipt_shape CHECK (
        (resume_source_execution_id IS NULL AND resume_requested_by IS NULL
         AND resume_requested_at IS NULL AND resume_transaction_id IS NULL)
        OR (resume_source_execution_id IS NOT NULL AND resume_requested_by IS NOT NULL
            AND resume_requested_at IS NOT NULL AND resume_transaction_id IS NOT NULL));

CREATE INDEX stage_execution_transition_run_idx
    ON stage_executions(run_id, transition_decided_at, stage_execution_id)
    WHERE transition_action IS NOT NULL;

-- Lifecycle writes may still update other columns after a decision is set.
CREATE FUNCTION contractor_protect_stage_transition() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF OLD.transition_action IS NOT NULL AND
       ROW(NEW.transition_action, NEW.transition_target_stage_name, NEW.transition_target_execution_id,
           NEW.transition_escalation_ordinal, NEW.transition_escalation_exhausted, NEW.transition_decided_at)
       IS DISTINCT FROM
       ROW(OLD.transition_action, OLD.transition_target_stage_name, OLD.transition_target_execution_id,
           OLD.transition_escalation_ordinal, OLD.transition_escalation_exhausted, OLD.transition_decided_at)
    THEN
        RAISE EXCEPTION 'Stage transition decisions are immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER stage_executions_transition_immutable
    BEFORE UPDATE OF transition_action, transition_target_stage_name, transition_target_execution_id,
        transition_escalation_ordinal, transition_escalation_exhausted, transition_decided_at
    ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_transition();

-- A receipt cannot be attached to an old Stage or borrow another transaction's
-- authority. Exact source snapshot validation remains in the existing owner.
DROP TRIGGER stage_executions_manual_continuation ON stage_executions;
CREATE OR REPLACE FUNCTION contractor_validate_manual_stage_execution() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF ROW(NEW.resume_source_execution_id, NEW.resume_requested_by, NEW.resume_requested_at, NEW.resume_transaction_id)
           IS DISTINCT FROM
           ROW(OLD.resume_source_execution_id, OLD.resume_requested_by, OLD.resume_requested_at, OLD.resume_transaction_id) THEN
            RAISE EXCEPTION 'manual continuation identity is immutable' USING ERRCODE = '23514';
        END IF;
        RETURN NEW;
    END IF;
    IF NEW.resume_source_execution_id IS NOT NULL AND (
        NEW.resume_transaction_id IS DISTINCT FROM pg_current_xact_id()
        OR NOT EXISTS (SELECT 1 FROM workflow_runs WHERE run_id = NEW.run_id AND owner_id = NEW.resume_requested_by)
    ) THEN
        RAISE EXCEPTION 'manual continuation receipt must be created by its owner in this transaction' USING ERRCODE = '23514';
    END IF;
    IF NEW.resume_source_execution_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM stage_executions AS source
        WHERE source.stage_execution_id = NEW.resume_source_execution_id
          AND source.run_id = NEW.run_id AND source.stage_name = NEW.stage_name
          AND source.attempt + 1 = NEW.attempt
          AND source.state IN ('failed', 'interrupted')
          AND source.execution_config_variant = NEW.execution_config_variant
          AND source.escalation_ordinal IS NOT DISTINCT FROM NEW.escalation_ordinal
          AND source.stage_spec_schema_version = NEW.stage_spec_schema_version
          AND source.stage_spec_snapshot = NEW.stage_spec_snapshot
          AND source.stage_context_schema_version = NEW.stage_context_schema_version
          AND source.stage_context_snapshot = NEW.stage_context_snapshot
    ) THEN
        RAISE EXCEPTION 'manual continuation must copy its terminal source' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER stage_executions_manual_continuation
    BEFORE INSERT OR UPDATE OF resume_source_execution_id, resume_requested_by,
        resume_requested_at, resume_transaction_id
    ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_validate_manual_stage_execution();

CREATE OR REPLACE FUNCTION contractor_protect_artifact_binding() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
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
                SELECT 1 FROM stage_executions AS target
                JOIN workflow_runs AS run ON run.run_id = target.run_id
                WHERE target.run_id = OLD.scope_id
                  AND target.resume_source_execution_id IS NOT NULL
                  AND target.resume_transaction_id = pg_current_xact_id()
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
