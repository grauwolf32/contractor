package runstore

// SQL statements for planner_store.go.

// appendPlannerEventSQL appends event $3 to planner session $2, locked FOR
// UPDATE and required to match StageExecution $12, invocation $13 and
// next_event_sequence $3. A non-empty $14 must be the Run's unexpired scheduler
// claim. It writes the Run event and linked planner event, then stores the new
// session state. Returns (session found, appended).
// Used by PostgresStore.AppendPlannerEvent.
var appendPlannerEventSQL = `
WITH locked AS (
    SELECT session.session_id, session.next_event_sequence, execution.run_id
    FROM planner_sessions AS session
    JOIN stage_executions AS execution
      ON execution.stage_execution_id = session.stage_execution_id
    WHERE session.session_id = $2
      AND execution.stage_execution_id = $12
      AND session.invocation_id = $13
    FOR UPDATE OF session
), owning AS (
    SELECT session_id, run_id
    FROM locked
    WHERE next_event_sequence = $3
), allocated AS (
    UPDATE workflow_runs AS run
    SET next_run_event_sequence = next_run_event_sequence + 1
    FROM owning
    WHERE run.run_id = owning.run_id
      AND ($14::text = '' OR (run.scheduler_claim_id = $14
           AND run.scheduler_claim_expires_at > clock_timestamp()))
    RETURNING run.run_id, run.next_run_event_sequence - 1 AS sequence_number
), inserted_run_event AS (
    INSERT INTO workflow_run_events (
        run_id, sequence_number, event_id, event_schema_version, kind, data
    )
    SELECT run_id, sequence_number, $8, $9, $10, $11::jsonb
    FROM allocated
    RETURNING run_id, sequence_number
), inserted AS (
    INSERT INTO planner_events (
        event_id, session_id, sequence_number, event_schema_version, event,
        run_id, run_event_sequence
    )
    SELECT $1, owning.session_id, $3, $4, $5::jsonb,
           inserted_run_event.run_id, inserted_run_event.sequence_number
    FROM owning CROSS JOIN inserted_run_event
    RETURNING session_id
), updated AS (
    UPDATE planner_sessions AS session
    SET state_schema_version = $6,
        state = $7::jsonb,
        next_event_sequence = next_event_sequence + 1,
        updated_at = clock_timestamp()
    FROM inserted
    WHERE session.session_id = inserted.session_id
    RETURNING session.session_id
)
SELECT EXISTS (SELECT 1 FROM locked), EXISTS (SELECT 1 FROM updated)`
