ALTER TABLE workflow_runs
    ADD COLUMN run_event_generation text NOT NULL
        DEFAULT ('events-' || md5(random()::text || clock_timestamp()::text))
        CHECK (btrim(run_event_generation) <> ''),
    ADD COLUMN next_run_event_sequence bigint NOT NULL DEFAULT 1
        CHECK (next_run_event_sequence > 0);

ALTER TABLE planner_sessions
    ADD COLUMN next_event_sequence bigint;

UPDATE planner_sessions AS session
SET next_event_sequence = COALESCE((
    SELECT max(event.sequence_number) + 1
    FROM planner_events AS event
    WHERE event.session_id = session.session_id
), 1);

ALTER TABLE planner_sessions
    ALTER COLUMN next_event_sequence SET NOT NULL,
    ALTER COLUMN next_event_sequence SET DEFAULT 1,
    ADD CONSTRAINT planner_session_next_event_sequence_positive
        CHECK (next_event_sequence > 0),
    ADD CONSTRAINT planner_session_state_size
        CHECK (octet_length(state::text) <= 2097152);

ALTER TABLE planner_events
    ADD CONSTRAINT planner_event_size
        CHECK (octet_length(event::text) <= 2097152);

CREATE TABLE workflow_run_events (
    run_id text NOT NULL REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
    sequence_number bigint NOT NULL CHECK (sequence_number > 0),
    event_id text NOT NULL UNIQUE CHECK (btrim(event_id) <> ''),
    event_schema_version text NOT NULL CHECK (btrim(event_schema_version) <> ''),
    kind text NOT NULL CHECK (kind IN (
        'planner.started',
        'planner.request_recorded',
        'planner.activity',
        'planner.plan_changed',
        'planner.current_changed',
        'planner.dispatch_selected',
        'planner.dispatch_started',
        'planner.dispatch_completed',
        'planner.finish_requested',
        'planner.completed',
        'planner.failed'
    )),
    data jsonb NOT NULL CHECK (
        jsonb_typeof(data) = 'object'
        AND octet_length(data::text) <= 2097152
    ),
    occurred_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (run_id, sequence_number)
);

CREATE INDEX workflow_run_events_time_idx
    ON workflow_run_events (run_id, occurred_at, sequence_number);

ALTER TABLE planner_events
    ADD COLUMN run_id text,
    ADD COLUMN run_event_sequence bigint,
    ADD CONSTRAINT planner_event_run_identity_shape CHECK (
        (run_id IS NULL AND run_event_sequence IS NULL)
        OR (run_id IS NOT NULL AND run_event_sequence IS NOT NULL)
    ),
    ADD CONSTRAINT planner_event_run_event
        FOREIGN KEY (run_id, run_event_sequence)
        REFERENCES workflow_run_events(run_id, sequence_number)
        ON DELETE CASCADE;

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
        OR NEW.request_idempotency_key IS DISTINCT FROM OLD.request_idempotency_key
        OR NEW.request_digest IS DISTINCT FROM OLD.request_digest
        OR NEW.run_event_generation IS DISTINCT FROM OLD.run_event_generation
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'immutable WorkflowRun fields cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.run_cancellation IS NOT NULL
        AND (
            NEW.cancellation_schema_version IS DISTINCT FROM OLD.cancellation_schema_version
            OR NEW.run_cancellation IS DISTINCT FROM OLD.run_cancellation
        )
    THEN
        RAISE EXCEPTION 'WorkflowRun cancellation cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION contractor_protect_planner_event_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'Planner events are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER planner_events_protect_immutable
BEFORE UPDATE ON planner_events
FOR EACH ROW EXECUTE FUNCTION contractor_protect_planner_event_immutable();

CREATE OR REPLACE FUNCTION contractor_protect_workflow_run_event_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'WorkflowRun events are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER workflow_run_events_protect_immutable
BEFORE UPDATE ON workflow_run_events
FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_event_immutable();
