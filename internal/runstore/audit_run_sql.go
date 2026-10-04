package runstore

// SQL statements for audit_run.go.

// createAuditRunSQL inserts an 'initializing' Audit-managed WorkflowRun bound
// to AuditExecution $12 and submission key $13, plus its metadata labels from
// the $14 JSON object. A deferred constraint requires the AuditExecution to
// point back to the Run before commit. The caller appends the column list and
// "FROM inserted_run". Used by PostgresStore.CreateAuditRun.
var createAuditRunSQL = `
WITH inserted_run AS (
INSERT INTO workflow_runs (
    run_id, owner_id, project_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    runtime_labels, runtime_config_snapshot, project_http_target_snapshot,
    publication_mode, audit_execution_id, audit_submission_key,
    state, state_reason_code, state_reason_message
) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10::jsonb, $11::jsonb,
          'audit-managed', $12, $13, 'initializing', 'created', '')
RETURNING *
), inserted_labels AS (
    INSERT INTO workflow_run_metadata_labels (run_id, ordinal, label_key, label_value)
    SELECT inserted_run.run_id,
           row_number() OVER (ORDER BY entry.key), entry.key, entry.value
    FROM inserted_run
    CROSS JOIN LATERAL jsonb_each_text($14::jsonb) AS entry
)
SELECT `
