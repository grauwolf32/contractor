package runstore

// SQL statements for stage_admission_store.go.

// markStageAdmittedSQL sets admitted_at of StageExecution $1 on its first
// admission, keeps it on a later one, and returns the Stage.
// Used by PostgresStore.MarkStageAdmitted.
var markStageAdmittedSQL = `
UPDATE stage_executions
SET admitted_at = COALESCE(admitted_at, clock_timestamp())
WHERE stage_execution_id = $1
RETURNING ` + stageExecutionColumns
