CREATE TABLE workflow_runs (
    run_id text PRIMARY KEY CHECK (btrim(run_id) <> ''),
    owner_id text NOT NULL CHECK (btrim(owner_id) <> ''),
    workflow_name text NOT NULL CHECK (btrim(workflow_name) <> ''),
    workflow_version text NOT NULL CHECK (btrim(workflow_version) <> ''),
    workflow_schema_version text NOT NULL CHECK (btrim(workflow_schema_version) <> ''),
    workflow_snapshot jsonb NOT NULL CHECK (jsonb_typeof(workflow_snapshot) = 'object'),
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(parameters) = 'object'),
    state text NOT NULL CHECK (state IN (
        'initializing', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled'
    )),
    state_reason_code text NOT NULL CHECK (btrim(state_reason_code) <> ''),
    state_reason_message text NOT NULL DEFAULT '',
    scheduler_claim_id text,
    scheduler_claimed_at timestamptz,
    scheduler_claim_expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    started_at timestamptz,
    finished_at timestamptz,
    CONSTRAINT workflow_run_claim_shape CHECK (
        (scheduler_claim_id IS NULL AND scheduler_claimed_at IS NULL AND scheduler_claim_expires_at IS NULL)
        OR
        (scheduler_claim_id IS NOT NULL AND btrim(scheduler_claim_id) <> '' AND scheduler_claimed_at IS NOT NULL
            AND scheduler_claim_expires_at > scheduler_claimed_at)
    ),
    CONSTRAINT workflow_run_terminal_shape CHECK (
        (state IN ('succeeded', 'failed', 'cancelled')) = (finished_at IS NOT NULL)
    ),
    CONSTRAINT workflow_run_terminal_unclaimed CHECK (
        state NOT IN ('succeeded', 'failed', 'cancelled') OR scheduler_claim_id IS NULL
    )
);

CREATE INDEX workflow_runs_owner_created_idx ON workflow_runs (owner_id, created_at, run_id);
CREATE INDEX workflow_runs_scheduler_idx ON workflow_runs (state, scheduler_claim_expires_at, created_at)
    WHERE state = 'running';

CREATE FUNCTION contractor_valid_stage_result(payload jsonb)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT COALESCE(
        jsonb_typeof(payload) = 'object'
        AND jsonb_typeof(payload->'apiVersion') = 'string'
        AND btrim(payload->>'apiVersion') <> ''
        AND payload->>'outcome' IN ('succeeded', 'failed')
        AND jsonb_typeof(payload->'summary') = 'string'
        AND btrim(payload->>'summary') <> ''
        AND jsonb_typeof(payload->'artifacts') = 'object'
        AND (
            (payload->>'outcome' = 'succeeded'
                AND (NOT (payload ? 'error') OR payload->'error' = 'null'::jsonb))
            OR
            (payload->>'outcome' = 'failed'
                AND jsonb_typeof(payload->'error') = 'object'
                AND jsonb_typeof(payload->'error'->'code') = 'string'
                AND btrim(payload->'error'->>'code') <> ''
                AND jsonb_typeof(payload->'error'->'message') = 'string'
                AND btrim(payload->'error'->>'message') <> ''
                AND jsonb_typeof(payload->'error'->'retryable') = 'boolean')
        ),
        false
    )
$$;

CREATE FUNCTION contractor_valid_stage_termination(payload jsonb)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT COALESCE(
        jsonb_typeof(payload) = 'object'
        AND jsonb_typeof(payload->'outcome') = 'string'
        AND payload->>'outcome' IN ('cancelled', 'interrupted')
        AND jsonb_typeof(payload->'code') = 'string'
        AND btrim(payload->>'code') <> ''
        AND jsonb_typeof(payload->'message') = 'string'
        AND btrim(payload->>'message') <> ''
        AND jsonb_typeof(payload->'retryable') = 'boolean'
        AND jsonb_typeof(payload->'phase') = 'string'
        AND payload->>'phase' IN ('preparing', 'running')
        AND jsonb_typeof(payload->'occurredAt') = 'string'
        AND btrim(payload->>'occurredAt') <> '',
        false
    )
$$;

CREATE TABLE stage_executions (
    stage_execution_id text PRIMARY KEY CHECK (btrim(stage_execution_id) <> ''),
    run_id text NOT NULL REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
    stage_name text NOT NULL CHECK (btrim(stage_name) <> ''),
    attempt integer NOT NULL CHECK (attempt > 0),
    previous_execution_id text,
    stage_spec_schema_version text NOT NULL CHECK (btrim(stage_spec_schema_version) <> ''),
    stage_spec_snapshot jsonb NOT NULL CHECK (jsonb_typeof(stage_spec_snapshot) = 'object'),
    stage_context_schema_version text NOT NULL CHECK (btrim(stage_context_schema_version) <> ''),
    stage_context_snapshot jsonb NOT NULL CHECK (jsonb_typeof(stage_context_snapshot) = 'object'),
    state text NOT NULL CHECK (state IN (
        'preparing', 'running', 'finalizing', 'aborting',
        'succeeded', 'failed', 'interrupted', 'cancelled'
    )),
    state_reason_code text NOT NULL CHECK (btrim(state_reason_code) <> ''),
    state_reason_message text NOT NULL DEFAULT '',
    planner_session_id text,
    planner_invocation_id text,
    candidate_result_schema_version text,
    candidate_stage_result jsonb,
    accepted_result_schema_version text,
    accepted_stage_result jsonb,
    termination_schema_version text,
    stage_termination jsonb,
    finalization_id text,
    finalization_deadline timestamptz,
    abort_id text,
    abort_deadline timestamptz,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    planner_started_at timestamptz,
    terminal_at timestamptz,
    CONSTRAINT stage_attempt_identity UNIQUE (run_id, stage_name, attempt),
    CONSTRAINT stage_execution_lineage_key UNIQUE (run_id, stage_name, stage_execution_id),
    CONSTRAINT stage_execution_lineage FOREIGN KEY (run_id, stage_name, previous_execution_id)
        REFERENCES stage_executions(run_id, stage_name, stage_execution_id)
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT stage_attempt_lineage_shape CHECK (
        (attempt = 1 AND previous_execution_id IS NULL)
        OR (attempt > 1 AND previous_execution_id IS NOT NULL)
    ),
    CONSTRAINT stage_planner_ids_shape CHECK (
        (planner_session_id IS NULL) = (planner_invocation_id IS NULL)
    ),
    CONSTRAINT stage_candidate_shape CHECK (
        (candidate_result_schema_version IS NULL) = (candidate_stage_result IS NULL)
        AND (candidate_stage_result IS NULL OR (
            btrim(candidate_result_schema_version) <> ''
            AND contractor_valid_stage_result(candidate_stage_result)
        ))
    ),
    CONSTRAINT stage_accepted_shape CHECK (
        (accepted_result_schema_version IS NULL) = (accepted_stage_result IS NULL)
        AND (accepted_stage_result IS NULL OR (
            btrim(accepted_result_schema_version) <> ''
            AND contractor_valid_stage_result(accepted_stage_result)
        ))
    ),
    CONSTRAINT stage_termination_payload_shape CHECK (
        (termination_schema_version IS NULL) = (stage_termination IS NULL)
        AND (stage_termination IS NULL OR (
            btrim(termination_schema_version) <> ''
            AND contractor_valid_stage_termination(stage_termination)
        ))
    ),
    CONSTRAINT stage_execution_state_shape CHECK (
        (
            state = 'preparing'
            AND planner_session_id IS NULL
            AND candidate_stage_result IS NULL
            AND accepted_stage_result IS NULL
            AND stage_termination IS NULL
            AND finalization_id IS NULL AND finalization_deadline IS NULL
            AND abort_id IS NULL AND abort_deadline IS NULL
            AND terminal_at IS NULL
        ) OR (
            state = 'running'
            AND planner_session_id IS NOT NULL
            AND candidate_stage_result IS NULL
            AND accepted_stage_result IS NULL
            AND stage_termination IS NULL
            AND finalization_id IS NULL AND finalization_deadline IS NULL
            AND abort_id IS NULL AND abort_deadline IS NULL
            AND planner_started_at IS NOT NULL AND terminal_at IS NULL
        ) OR (
            state = 'finalizing'
            AND planner_session_id IS NOT NULL
            AND candidate_stage_result IS NOT NULL
            AND accepted_stage_result IS NULL
            AND stage_termination IS NULL
            AND finalization_id IS NOT NULL AND btrim(finalization_id) <> '' AND finalization_deadline IS NOT NULL
            AND abort_id IS NULL AND abort_deadline IS NULL
            AND planner_started_at IS NOT NULL AND terminal_at IS NULL
        ) OR (
            state = 'aborting'
            AND candidate_stage_result IS NULL
            AND accepted_stage_result IS NULL
            AND stage_termination IS NOT NULL
            AND finalization_id IS NULL AND finalization_deadline IS NULL
            AND abort_id IS NOT NULL AND btrim(abort_id) <> '' AND abort_deadline IS NOT NULL
            AND terminal_at IS NULL
            AND (
                (stage_termination->>'phase' = 'preparing' AND planner_session_id IS NULL)
                OR
                (stage_termination->>'phase' = 'running'
                    AND planner_session_id IS NOT NULL AND planner_started_at IS NOT NULL)
            )
        ) OR (
            state IN ('succeeded', 'failed')
            AND planner_session_id IS NOT NULL
            AND candidate_stage_result IS NOT NULL
            AND accepted_stage_result = candidate_stage_result
            AND accepted_result_schema_version = candidate_result_schema_version
            AND accepted_stage_result->>'outcome' = state
            AND stage_termination IS NULL
            AND finalization_id IS NOT NULL AND btrim(finalization_id) <> '' AND finalization_deadline IS NOT NULL
            AND abort_id IS NULL AND abort_deadline IS NULL
            AND planner_started_at IS NOT NULL AND terminal_at IS NOT NULL
        ) OR (
            state IN ('interrupted', 'cancelled')
            AND candidate_stage_result IS NULL
            AND accepted_stage_result IS NULL
            AND stage_termination IS NOT NULL
            AND stage_termination->>'outcome' = state
            AND finalization_id IS NULL AND finalization_deadline IS NULL
            AND abort_id IS NOT NULL AND btrim(abort_id) <> '' AND abort_deadline IS NOT NULL
            AND terminal_at IS NOT NULL
            AND (
                (stage_termination->>'phase' = 'preparing' AND planner_session_id IS NULL)
                OR
                (stage_termination->>'phase' = 'running'
                    AND planner_session_id IS NOT NULL AND planner_started_at IS NOT NULL)
            )
        )
    )
);

CREATE INDEX stage_executions_run_created_idx ON stage_executions (run_id, created_at, stage_execution_id);
CREATE INDEX stage_executions_recovery_idx ON stage_executions (state, updated_at)
    WHERE state IN ('preparing', 'running', 'finalizing', 'aborting');

CREATE TABLE stage_allocations (
    allocation_id text PRIMARY KEY CHECK (btrim(allocation_id) <> ''),
    stage_execution_id text NOT NULL REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE,
    logical_agent_name text NOT NULL CHECK (btrim(logical_agent_name) <> ''),
    namespace text NOT NULL CHECK (btrim(namespace) <> '' AND position('/' in namespace) = 0),
    agent_template_ref jsonb NOT NULL CHECK (jsonb_typeof(agent_template_ref) = 'object'),
    worker_runtime_ref jsonb NOT NULL CHECK (jsonb_typeof(worker_runtime_ref) = 'object'),
    runtime_agent_instance_id text NOT NULL CHECK (btrim(runtime_agent_instance_id) <> ''),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (stage_execution_id, logical_agent_name)
);

CREATE TABLE planner_sessions (
    session_id text PRIMARY KEY CHECK (btrim(session_id) <> ''),
    stage_execution_id text NOT NULL UNIQUE
        REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE,
    invocation_id text NOT NULL UNIQUE CHECK (btrim(invocation_id) <> ''),
    state_schema_version text NOT NULL CHECK (btrim(state_schema_version) <> ''),
    state jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(state) = 'object'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (stage_execution_id, session_id, invocation_id)
);

ALTER TABLE stage_executions
    ADD CONSTRAINT stage_execution_planner_session
    FOREIGN KEY (stage_execution_id, planner_session_id, planner_invocation_id)
    REFERENCES planner_sessions(stage_execution_id, session_id, invocation_id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE planner_events (
    event_id text PRIMARY KEY CHECK (btrim(event_id) <> ''),
    session_id text NOT NULL REFERENCES planner_sessions(session_id) ON DELETE CASCADE,
    sequence_number bigint NOT NULL CHECK (sequence_number > 0),
    event_schema_version text NOT NULL CHECK (btrim(event_schema_version) <> ''),
    event jsonb NOT NULL CHECK (jsonb_typeof(event) = 'object'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (session_id, sequence_number)
);

CREATE OR REPLACE FUNCTION contractor_protect_workflow_run_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.run_id IS DISTINCT FROM OLD.run_id
        OR NEW.owner_id IS DISTINCT FROM OLD.owner_id
        OR NEW.workflow_name IS DISTINCT FROM OLD.workflow_name
        OR NEW.workflow_version IS DISTINCT FROM OLD.workflow_version
        OR NEW.workflow_schema_version IS DISTINCT FROM OLD.workflow_schema_version
        OR NEW.workflow_snapshot IS DISTINCT FROM OLD.workflow_snapshot
        OR NEW.parameters IS DISTINCT FROM OLD.parameters
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'immutable WorkflowRun fields cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER workflow_runs_protect_immutable
BEFORE UPDATE ON workflow_runs
FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_immutable();

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

CREATE TRIGGER stage_executions_protect_immutable
BEFORE UPDATE ON stage_executions
FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_execution_immutable();
