package runstore

// SQL statements for stage_store.go.

// hasSummaryEscalationSQL probes the same predicate as the unique strategy
// root index. Used by PostgresStore.HasSummaryEscalation.
var hasSummaryEscalationSQL = `
SELECT EXISTS (
    SELECT 1 FROM stage_executions
    WHERE run_id=$1 AND stage_name=$2
      AND execution_config_variant='summary_escalation'
      AND resume_source_execution_id IS NULL
      AND stage_context_snapshot->'continuation'->>'sourceStageExecutionId'=previous_execution_id
)`

// createStageExecutionSQL inserts StageExecution $1 of Run $2, attempt $4 of
// Stage $3 after previous execution $5, with execution configuration variant
// $6, escalation ordinal $7 and resume source $8, StageSpec $10 of schema
// version $9 and StageContext $12 of schema version $11, in state preparing
// with reason created, and returns it. The same statement sets the Run's
// scheduler_lane_class to $13, the class of the new StageExecution, which is
// the one the Scheduler progresses next (SCH-R175); a failed insert changes
// neither row. Used by PostgresStore.CreateStageExecution.
var createStageExecutionSQL = `
WITH inserted AS (
INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt, previous_execution_id,
    execution_config_variant, escalation_ordinal, resume_source_execution_id,
    stage_spec_schema_version, stage_spec_snapshot,
    stage_context_schema_version, stage_context_snapshot,
    state, state_reason_code, state_reason_message
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12::jsonb, 'preparing', 'created', '')
RETURNING *
), lane AS (
UPDATE workflow_runs AS run
   SET scheduler_lane_class = $13
  FROM inserted
 WHERE run.run_id = inserted.run_id
   AND run.scheduler_lane_class IS DISTINCT FROM $13
)
SELECT ` + stageExecutionColumns + ` FROM inserted`

// getStageExecutionSQL reads StageExecution $1.
// Used by PostgresStore.GetStageExecution.
var getStageExecutionSQL = `SELECT ` + stageExecutionColumns + ` FROM stage_executions WHERE stage_execution_id = $1`

// listTerminalStageExecutionsWithAllocationsSQL selects up to $1 terminal
// StageExecutions that still have unreleased allocations, ordered by their
// oldest release attempt (never attempted first) so retries rotate fairly.
// It is a prefix: the caller appends the column list and the join back to
// stage_executions.
// Used by PostgresStore.ListTerminalStageExecutionsWithAllocations.
var listTerminalStageExecutionsWithAllocationsSQL = `
WITH pending AS (
    SELECT allocation.stage_execution_id,
           min(COALESCE(allocation.release_attempted_at, '-infinity'::timestamptz)) AS next_attempt_at
    FROM stage_allocations AS allocation
    JOIN stage_executions AS candidate
      ON candidate.stage_execution_id = allocation.stage_execution_id
    WHERE allocation.release_completed_at IS NULL
      AND candidate.state IN ('succeeded', 'failed', 'interrupted', 'cancelled')
    GROUP BY allocation.stage_execution_id
    ORDER BY next_attempt_at, allocation.stage_execution_id
    LIMIT $1
)
SELECT `

// getActiveStageExecutionSQL returns, oldest first, up to two StageExecutions
// of Run $1 that are preparing, running, finalizing or aborting; a second row
// means the Run has more than the one active Stage it may have.
// Used by PostgresStore.GetActiveStageExecution.
var getActiveStageExecutionSQL = `
SELECT ` + stageExecutionColumns + `
FROM stage_executions
WHERE run_id = $1 AND state IN ('preparing', 'running', 'finalizing', 'aborting')
ORDER BY created_at, stage_execution_id
LIMIT 2`

// listStageExecutionsSQL returns every StageExecution of Run $1, oldest
// first. Used by PostgresStore.ListStageExecutions.
var listStageExecutionsSQL = `
SELECT ` + stageExecutionColumns + `
FROM stage_executions
WHERE run_id = $1
ORDER BY created_at, stage_execution_id`

// startPlannerSQL moves StageExecution $1 from preparing to running (state
// CAS) and creates planner session $2. It reserves two Run event sequence
// numbers and appends a lifecycle.changed event and the planner-started event.
// Returns the session ID; no row means the Stage was not preparing.
// Used by PostgresStore.StartPlanner.
var startPlannerSQL = `
WITH transitioned AS (
    UPDATE stage_executions
    SET state = 'running',
        state_reason_code = $6,
        state_reason_message = $7,
        planner_session_id = $2,
        planner_invocation_id = $3,
        planner_started_at = clock_timestamp(),
        updated_at = clock_timestamp()
    WHERE stage_execution_id = $1 AND state = 'preparing'
    RETURNING stage_execution_id, run_id
), inserted_session AS (
    INSERT INTO planner_sessions (
        session_id, stage_execution_id, invocation_id, state_schema_version, state,
        next_event_sequence
    )
    SELECT $2, stage_execution_id, $3, $4, $5::jsonb, 2
    FROM transitioned
    RETURNING session_id
), allocated AS (
    UPDATE workflow_runs AS run
    SET next_run_event_sequence = next_run_event_sequence + 2
    FROM transitioned
    WHERE run.run_id = transitioned.run_id
    RETURNING run.run_id,
              run.next_run_event_sequence - 2 AS lifecycle_sequence,
              run.next_run_event_sequence - 1 AS planner_sequence
), inserted_lifecycle_event AS (
    INSERT INTO workflow_run_events (
        run_id, sequence_number, event_id, event_schema_version, kind, data
    )
    SELECT run_id,
           lifecycle_sequence,
           'lifecycle-' || md5(
               random()::text || clock_timestamp()::text || run_id || lifecycle_sequence::text
           ),
           'contractor/v1alpha1',
           'lifecycle.changed',
           jsonb_build_object(
               'runId', run_id,
               'resource', 'stageExecution',
               'stageExecutionId', $1::text,
               'state', 'running'
           )
    FROM allocated
    RETURNING run_id, sequence_number
), inserted_event AS (
    INSERT INTO workflow_run_events (
        run_id, sequence_number, event_id, event_schema_version, kind, data
    )
    SELECT run_id, planner_sequence, $8, $9, $10, $11::jsonb
    FROM allocated
    RETURNING run_id, sequence_number
)
SELECT inserted_session.session_id
FROM inserted_session CROSS JOIN inserted_lifecycle_event CROSS JOIN inserted_event`

// enterFinalizingSQL moves StageExecution $1 from running to finalizing (state
// CAS) and stores the candidate StageResult, finalization ID and deadline.
// The caller requires exactly one affected row.
// Used by PostgresStore.EnterFinalizing.
var enterFinalizingSQL = `
UPDATE stage_executions
SET state = 'finalizing',
    state_reason_code = $6,
    state_reason_message = $7,
    candidate_result_schema_version = $2,
    candidate_stage_result = $3::jsonb,
    finalization_id = $4,
    finalization_deadline = $5,
    updated_at = clock_timestamp()
WHERE stage_execution_id = $1 AND state = 'running'`

// completeStageResultSQL makes a finalizing StageExecution terminal with
// outcome $4, accepting the result only if it equals the stored candidate
// (same schema version and JSON). Sets terminal_at; the caller requires
// exactly one affected row. Used by PostgresStore.CompleteStageResult.
var completeStageResultSQL = `
UPDATE stage_executions
SET state = $4,
    state_reason_code = 'result_accepted',
    state_reason_message = '',
    accepted_result_schema_version = $2,
    accepted_stage_result = $3::jsonb,
    terminal_at = clock_timestamp(),
    updated_at = clock_timestamp()
WHERE stage_execution_id = $1
  AND state = 'finalizing'
  AND candidate_result_schema_version = $2
  AND candidate_stage_result = $3::jsonb`

// enterAbortingSQL moves StageExecution $1 from expected state $2 (preparing
// or running) to aborting and stores the StageTermination, abort ID and
// deadline. The caller requires exactly one affected row.
// Used by PostgresStore.EnterAborting.
var enterAbortingSQL = `
UPDATE stage_executions
SET state = 'aborting',
    state_reason_code = $7,
    state_reason_message = $8,
    termination_schema_version = $3,
    stage_termination = $4::jsonb,
    abort_id = $5,
    abort_deadline = $6,
    updated_at = clock_timestamp()
WHERE stage_execution_id = $1 AND state = $2`
