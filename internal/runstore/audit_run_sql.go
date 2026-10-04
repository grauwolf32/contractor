package runstore

// SQL statements for audit_run.go.

// createAuditRunSQL inserts an 'initializing' Audit-managed WorkflowRun bound
// to AuditExecution $12 and submission key $13, with its metadata labels
// from the $14 JSON object. A deferred constraint requires the AuditExecution to
// point back to the Run before commit. The caller appends the column list and
// "FROM inserted_run". Used by PostgresStore.CreateAuditRun.
var createAuditRunSQL = `
WITH inserted_run AS (
INSERT INTO workflow_runs (
    run_id, owner_id, project_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    runtime_labels, runtime_config_snapshot, project_http_target_snapshot,
    publication_mode, audit_execution_id, audit_submission_key, metadata_labels,
    state, state_reason_code, state_reason_message
) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10::jsonb, $11::jsonb,
          'audit-managed', $12, $13, $14::jsonb, 'initializing', 'created', '')
RETURNING *
)
SELECT `
