package scheduler

// SQL statements for postgres.go.

// failRunWithActiveStagesSQL makes every preparing or running StageExecution
// of Run $1 terminal at once, skipping the aborting phase: the state becomes
// the outcome of StageTermination $2, stored with its phase set to the Stage's
// prior state, abort ID $4, and abort deadline and terminal_at set to now.
// Used by PostgresPersistence.FailRunWithActiveStages.
var failRunWithActiveStagesSQL = `
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
