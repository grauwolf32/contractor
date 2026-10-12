package runstore

// SQL statements for run_skills.go.

// lockRunSkillInitializationSQL reads WorkflowRun $1 and locks its row until
// the transaction ends. Used by PostgresStore.LockRunSkillInitialization.
var lockRunSkillInitializationSQL = `
SELECT ` + workflowRunColumns + `
FROM workflow_runs
WHERE run_id = $1
	FOR UPDATE`
