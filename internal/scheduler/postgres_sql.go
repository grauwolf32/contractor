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

// verifyRequiredWorkflowOutputsSQL returns the media type of the Artifact
// version at the current revision of Run $1's binding $2 in the 'outputs'
// namespace. No row means the output is not bound.
// Used by verifyRequiredWorkflowOutputs.
var verifyRequiredWorkflowOutputsSQL = `
SELECT version.media_type
FROM artifact_bindings AS binding
JOIN artifact_binding_revisions AS revision
  ON revision.scope_kind = binding.scope_kind
 AND revision.scope_id = binding.scope_id
 AND revision.namespace = binding.namespace
 AND revision.name = binding.name
 AND revision.revision = binding.current_revision
JOIN artifact_versions AS version ON version.version_id = revision.version_id
WHERE binding.scope_kind = 'run' AND binding.scope_id = $1
  AND binding.namespace = 'outputs' AND binding.name = $2`
