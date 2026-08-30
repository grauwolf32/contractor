ALTER TABLE stage_executions
    ADD COLUMN execution_config_variant text NOT NULL DEFAULT 'base',
    ADD COLUMN escalation_ordinal integer,
    ADD CONSTRAINT stage_execution_config_variant_shape CHECK (
        (
            execution_config_variant = 'base'
            AND escalation_ordinal IS NULL
        ) OR (
            execution_config_variant IN ('failed_escalation', 'interrupted_escalation')
            AND escalation_ordinal IS NOT NULL
            AND escalation_ordinal > 0
        )
    );

CREATE UNIQUE INDEX stage_execution_escalation_ordinal_unique
    ON stage_executions (run_id, stage_name, execution_config_variant, escalation_ordinal)
    WHERE execution_config_variant <> 'base';

ALTER TABLE stage_transition_decisions
    DROP CONSTRAINT stage_transition_decisions_action_check,
    DROP CONSTRAINT stage_transition_decision_shape,
    ADD COLUMN escalation_ordinal integer,
    ADD COLUMN escalation_exhausted boolean NOT NULL DEFAULT false,
    ADD CONSTRAINT stage_transition_action CHECK (
        action IN ('next', 'retry', 'escalate', 'succeed', 'fail')
    ),
    ADD CONSTRAINT stage_transition_decision_shape CHECK (
        (
            action IN ('next', 'retry', 'escalate')
            AND target_stage_name IS NOT NULL AND btrim(target_stage_name) <> ''
            AND target_execution_id IS NOT NULL AND btrim(target_execution_id) <> ''
        ) OR (
            action IN ('succeed', 'fail')
            AND target_stage_name IS NULL
            AND target_execution_id IS NULL
        )
    ),
    ADD CONSTRAINT stage_transition_escalation_shape CHECK (
        (
            action = 'escalate'
            AND escalation_ordinal IS NOT NULL AND escalation_ordinal > 0
            AND escalation_exhausted = false
        ) OR (
            action IN ('next', 'fail')
            AND escalation_ordinal IS NOT NULL AND escalation_ordinal > 0
            AND escalation_exhausted = true
        ) OR (
            action <> 'escalate'
            AND escalation_ordinal IS NULL
            AND escalation_exhausted = false
        )
    );

CREATE OR REPLACE FUNCTION contractor_protect_stage_execution_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.stage_execution_id IS DISTINCT FROM OLD.stage_execution_id
        OR NEW.run_id IS DISTINCT FROM OLD.run_id
        OR NEW.stage_name IS DISTINCT FROM OLD.stage_name
        OR NEW.attempt IS DISTINCT FROM OLD.attempt
        OR NEW.previous_execution_id IS DISTINCT FROM OLD.previous_execution_id
        OR NEW.execution_config_variant IS DISTINCT FROM OLD.execution_config_variant
        OR NEW.escalation_ordinal IS DISTINCT FROM OLD.escalation_ordinal
        OR NEW.stage_spec_schema_version IS DISTINCT FROM OLD.stage_spec_schema_version
        OR NEW.stage_spec_snapshot IS DISTINCT FROM OLD.stage_spec_snapshot
        OR NEW.stage_context_schema_version IS DISTINCT FROM OLD.stage_context_schema_version
        OR NEW.stage_context_snapshot IS DISTINCT FROM OLD.stage_context_snapshot
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'immutable StageExecution fields cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.planner_session_id IS NOT NULL
        AND (NEW.planner_session_id IS DISTINCT FROM OLD.planner_session_id
            OR NEW.planner_invocation_id IS DISTINCT FROM OLD.planner_invocation_id)
    THEN
        RAISE EXCEPTION 'Planner identity cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.candidate_stage_result IS NOT NULL
        AND (NEW.candidate_stage_result IS DISTINCT FROM OLD.candidate_stage_result
            OR NEW.candidate_result_schema_version IS DISTINCT FROM OLD.candidate_result_schema_version)
    THEN
        RAISE EXCEPTION 'candidate StageResult cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.accepted_stage_result IS NOT NULL
        AND (NEW.accepted_stage_result IS DISTINCT FROM OLD.accepted_stage_result
            OR NEW.accepted_result_schema_version IS DISTINCT FROM OLD.accepted_result_schema_version)
    THEN
        RAISE EXCEPTION 'accepted StageResult cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.stage_termination IS NOT NULL
        AND (NEW.stage_termination IS DISTINCT FROM OLD.stage_termination
            OR NEW.termination_schema_version IS DISTINCT FROM OLD.termination_schema_version)
    THEN
        RAISE EXCEPTION 'StageTermination cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.finalization_id IS NOT NULL
        AND (NEW.finalization_id IS DISTINCT FROM OLD.finalization_id
            OR NEW.finalization_deadline IS DISTINCT FROM OLD.finalization_deadline)
    THEN
        RAISE EXCEPTION 'finalization identity cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.abort_id IS NOT NULL
        AND (NEW.abort_id IS DISTINCT FROM OLD.abort_id
            OR NEW.abort_deadline IS DISTINCT FROM OLD.abort_deadline)
    THEN
        RAISE EXCEPTION 'abort identity cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
