ALTER TABLE workflow_run_events
    DROP CONSTRAINT workflow_run_events_kind_check,
    ADD CONSTRAINT workflow_run_events_kind_check CHECK (kind IN (
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
        'planner.failed',
        'lifecycle.changed'
    ));

CREATE OR REPLACE FUNCTION contractor_append_workflow_run_lifecycle_event(
    target_run_id text,
    resource_kind text,
    target_stage_execution_id text,
    resource_state text
)
RETURNS bigint
LANGUAGE plpgsql
AS $$
DECLARE
    allocated_sequence bigint;
    lifecycle_data jsonb;
BEGIN
    IF resource_kind NOT IN ('run', 'stageExecution')
        OR btrim(resource_state) = ''
        OR (resource_kind = 'run' AND target_stage_execution_id IS NOT NULL)
        OR (
            resource_kind = 'stageExecution'
            AND (target_stage_execution_id IS NULL OR btrim(target_stage_execution_id) = '')
        )
    THEN
        RAISE EXCEPTION 'invalid WorkflowRun lifecycle event' USING ERRCODE = '23514';
    END IF;

    UPDATE workflow_runs
    SET next_run_event_sequence = next_run_event_sequence + 1
    WHERE run_id = target_run_id
    RETURNING next_run_event_sequence - 1 INTO allocated_sequence;

    IF allocated_sequence IS NULL THEN
        RAISE EXCEPTION 'WorkflowRun lifecycle owner is missing' USING ERRCODE = '23503';
    END IF;

    lifecycle_data := jsonb_build_object(
        'runId', target_run_id,
        'resource', resource_kind,
        'state', resource_state
    );
    IF target_stage_execution_id IS NOT NULL THEN
        lifecycle_data := lifecycle_data || jsonb_build_object(
            'stageExecutionId', target_stage_execution_id
        );
    END IF;

    INSERT INTO workflow_run_events (
        run_id, sequence_number, event_id, event_schema_version, kind, data
    ) VALUES (
        target_run_id,
        allocated_sequence,
        'lifecycle-' || md5(
            random()::text || clock_timestamp()::text || target_run_id || allocated_sequence::text
        ),
        'contractor/v1alpha1',
        'lifecycle.changed',
        lifecycle_data
    );
    RETURN allocated_sequence;
END;
$$;

CREATE OR REPLACE FUNCTION contractor_capture_workflow_run_lifecycle_event()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' OR OLD.state IS DISTINCT FROM NEW.state THEN
        PERFORM contractor_append_workflow_run_lifecycle_event(
            NEW.run_id, 'run', NULL, NEW.state
        );
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER workflow_runs_capture_lifecycle_event
AFTER INSERT OR UPDATE ON workflow_runs
FOR EACH ROW EXECUTE FUNCTION contractor_capture_workflow_run_lifecycle_event();

CREATE OR REPLACE FUNCTION contractor_capture_stage_execution_lifecycle_event()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM contractor_append_workflow_run_lifecycle_event(
            NEW.run_id, 'stageExecution', NEW.stage_execution_id, NEW.state
        );
        RETURN NEW;
    END IF;

    IF OLD.state IS DISTINCT FROM NEW.state THEN
        -- StartPlanner allocates this lifecycle event and planner.started in one
        -- explicit two-sequence CTE to avoid updating the owning Run twice in
        -- the same SQL statement.
        IF OLD.state = 'preparing'
            AND NEW.state = 'running'
            AND OLD.planner_session_id IS NULL
            AND NEW.planner_session_id IS NOT NULL
        THEN
            RETURN NEW;
        END IF;
        PERFORM contractor_append_workflow_run_lifecycle_event(
            NEW.run_id, 'stageExecution', NEW.stage_execution_id, NEW.state
        );
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER stage_executions_capture_lifecycle_event
AFTER INSERT OR UPDATE ON stage_executions
FOR EACH ROW EXECUTE FUNCTION contractor_capture_stage_execution_lifecycle_event();

CREATE OR REPLACE FUNCTION contractor_notify_workflow_run_event()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM pg_notify('contractor_run_events_v1', NEW.run_id);
    RETURN NEW;
END;
$$;

CREATE TRIGGER workflow_run_events_notify_after_commit
AFTER INSERT ON workflow_run_events
FOR EACH ROW EXECUTE FUNCTION contractor_notify_workflow_run_event();
