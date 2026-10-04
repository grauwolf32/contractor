package runstore

// SQL statements for planner_store.go.

// appendPlannerEventSQL appends the next event of planner session $1, locked
// FOR UPDATE and required to match StageExecution $9, invocation $10 and
// next_event_sequence $2. A non-empty $11 must be the Run's unexpired scheduler
// claim. It writes the Run event, then stores the new session state. Returns
// (session found, appended). Used by PostgresStore.AppendPlannerEvent.
var appendPlannerEventSQL = `
WITH locked AS (
    SELECT session.session_id, session.next_event_sequence, execution.run_id
    FROM planner_sessions AS session
    JOIN stage_executions AS execution
      ON execution.stage_execution_id = session.stage_execution_id
    WHERE session.session_id = $1
      AND execution.stage_execution_id = $9
      AND session.invocation_id = $10
    FOR UPDATE OF session
), owning AS (
    SELECT session_id, run_id
    FROM locked
    WHERE next_event_sequence = $2
), allocated AS (
    UPDATE workflow_runs AS run
    SET next_run_event_sequence = next_run_event_sequence + 1
    FROM owning
    WHERE run.run_id = owning.run_id
      AND ($11::text = '' OR (run.scheduler_claim_id = $11
           AND run.scheduler_claim_expires_at > clock_timestamp()))
    RETURNING run.run_id, run.next_run_event_sequence - 1 AS sequence_number
), inserted_run_event AS (
    INSERT INTO workflow_run_events (
        run_id, sequence_number, event_id, event_schema_version, kind, data
    )
    SELECT run_id, sequence_number, $5, $6, $7, $8::jsonb
    FROM allocated
    RETURNING run_id
), updated AS (
    UPDATE planner_sessions AS session
    SET state_schema_version = $3,
        state = $4::jsonb,
        next_event_sequence = next_event_sequence + 1,
        updated_at = clock_timestamp()
    FROM owning CROSS JOIN inserted_run_event
    WHERE session.session_id = owning.session_id
    RETURNING session.session_id
)
SELECT EXISTS (SELECT 1 FROM locked), EXISTS (SELECT 1 FROM updated)`
