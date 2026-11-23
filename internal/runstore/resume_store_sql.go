package runstore

// SQL statements for resume_store.go.

// resumableStageSQL returns the latest StageExecution of failed Run $1 (owner
// $2) if the Run can be resumed: ordinary, uncancelled and not Audit-bound,
// last Stage failed or interrupted, Project (if any) active and not an
// evaluation, no unreleased allocations, active Stages or output publications,
// and fewer than 1024 Stages. Advisory read without locks.
// Used by PostgresStore.ResumableStage.
var resumableStageSQL = `
SELECT execution.stage_execution_id
FROM workflow_runs AS run
JOIN LATERAL (
 SELECT stage_execution_id, state FROM stage_executions
 WHERE run_id = run.run_id ORDER BY created_at DESC, stage_execution_id DESC LIMIT 1
) AS execution ON true
WHERE run.run_id = $1 AND run.owner_id = $2 AND run.state = 'failed'
 AND run.publication_mode = 'ordinary' AND run.audit_execution_id IS NULL
 AND run.run_cancellation IS NULL
 AND execution.state IN ('failed', 'interrupted')
 AND (run.project_id IS NULL OR EXISTS (
   SELECT 1 FROM projects WHERE project_id = run.project_id
   AND lifecycle_state = 'active' AND kind <> 'evaluation'
 ))
 AND NOT EXISTS (SELECT 1 FROM stage_executions e JOIN stage_allocations a
   ON a.stage_execution_id = e.stage_execution_id
   WHERE e.run_id = run.run_id AND a.release_completed_at IS NULL)
 AND NOT EXISTS (SELECT 1 FROM stage_executions e WHERE e.run_id = run.run_id
   AND e.state IN ('preparing','running','finalizing','aborting'))
 AND NOT EXISTS (SELECT 1 FROM workflow_run_output_publications WHERE run_id = run.run_id)
 AND (SELECT count(*) FROM stage_executions WHERE run_id = run.run_id) < 1024`
