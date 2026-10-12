package runstore

// SQL statements for run_failure_store.go.

// terminateActiveStagesSQL makes every preparing or running StageExecution
// of Run $1 terminal at once, skipping the aborting phase: the state becomes
// the outcome of StageTermination $2, stored under schema version $3 with its
// phase set to the Stage's prior state, abort ID $4, and abort deadline and
// terminal_at set to now.
// Used by PostgresTxStore.FailRunWithActiveStages.
var terminateActiveStagesSQL = `
UPDATE stage_executions
SET state = $2::jsonb->>'outcome',
    state_reason_code = 'termination_committed',
    state_reason_message = '',
    termination_schema_version = $3,
    stage_termination = jsonb_set($2::jsonb, '{phase}', to_jsonb(state)),
    abort_id = $4,
    abort_deadline = clock_timestamp(),
    terminal_at = clock_timestamp(),
    updated_at = clock_timestamp()
WHERE run_id = $1 AND state IN ('preparing', 'running')`

// commitAbortingStagesSQL commits the stored StageTermination of every
// aborting StageExecution of Run $1, as CompleteStageTermination does for
// one Stage.
// Used by PostgresTxStore.FailRunWithActiveStages.
var commitAbortingStagesSQL = `
UPDATE stage_executions
SET state = stage_termination->>'outcome',
    state_reason_code = 'termination_committed',
    state_reason_message = '',
    terminal_at = clock_timestamp(),
    updated_at = clock_timestamp()
WHERE run_id = $1 AND state = 'aborting'`

// acceptFinalizingCandidatesSQL accepts the durable candidate StageResult of
// every finalizing StageExecution of Run $1 as its result, as cancellation
// does: a StageTermination cannot replace a durable candidate, and missing
// finalization reports stay incomplete.
// Used by PostgresTxStore.FailRunWithActiveStages.
var acceptFinalizingCandidatesSQL = `
UPDATE stage_executions
SET state = candidate_stage_result->>'outcome',
    state_reason_code = 'result_accepted',
    state_reason_message = '',
    accepted_result_schema_version = candidate_result_schema_version,
    accepted_stage_result = candidate_stage_result,
    terminal_at = clock_timestamp(),
    updated_at = clock_timestamp()
WHERE run_id = $1 AND state = 'finalizing'`
