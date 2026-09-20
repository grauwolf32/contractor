-- A manual continuation inherits an automatic escalation's configuration and
-- ordinal, but is not another automatic escalation budget position.
ALTER TABLE stage_executions ADD COLUMN resume_source_execution_id text;

UPDATE stage_executions AS target
SET resume_source_execution_id = receipt.source_execution_id
FROM run_stage_resumptions AS receipt
WHERE receipt.target_execution_id = target.stage_execution_id;

ALTER TABLE run_stage_resumptions
    ADD CONSTRAINT run_stage_resumption_target_source_unique
    UNIQUE (target_execution_id, source_execution_id);

-- The stage is inserted first and the receipt second in one transaction.
ALTER TABLE stage_executions
    ADD CONSTRAINT stage_execution_resume_receipt
    FOREIGN KEY (stage_execution_id, resume_source_execution_id)
    REFERENCES run_stage_resumptions(target_execution_id, source_execution_id)
    DEFERRABLE INITIALLY DEFERRED,
    ADD CONSTRAINT stage_execution_resume_previous CHECK (
        resume_source_execution_id IS NULL OR (
            previous_execution_id IS NOT NULL
            AND resume_source_execution_id = previous_execution_id
        )
    );

DROP INDEX stage_execution_escalation_ordinal_unique;
CREATE UNIQUE INDEX stage_execution_escalation_ordinal_unique
    ON stage_executions (run_id, stage_name, execution_config_variant, escalation_ordinal)
    WHERE execution_config_variant <> 'base' AND resume_source_execution_id IS NULL;

CREATE FUNCTION contractor_validate_manual_stage_execution()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF NEW.resume_source_execution_id IS DISTINCT FROM OLD.resume_source_execution_id THEN
            RAISE EXCEPTION 'manual continuation identity is immutable' USING ERRCODE = '23514';
        END IF;
        RETURN NEW;
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
BEFORE INSERT OR UPDATE ON stage_executions
FOR EACH ROW EXECUTE FUNCTION contractor_validate_manual_stage_execution();

CREATE FUNCTION contractor_validate_run_stage_resumption()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM stage_executions AS target
        WHERE target.stage_execution_id = NEW.target_execution_id
          AND target.run_id = NEW.run_id
          AND target.resume_source_execution_id = NEW.source_execution_id
    ) THEN
        RAISE EXCEPTION 'manual continuation receipt must match its target' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER run_stage_resumptions_validate
BEFORE INSERT ON run_stage_resumptions
FOR EACH ROW EXECUTE FUNCTION contractor_validate_run_stage_resumption();
