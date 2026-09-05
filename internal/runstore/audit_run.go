package runstore

import (
	"context"
	"encoding/json"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// CreateAuditRun inserts one initializing Run with immutable trusted Audit
// provenance. A deferred database constraint requires the matching
// AuditExecution to point back to this Run before the surrounding transaction
// commits, so this method cannot create a durable orphan on its own.
func (s *PostgresStore) CreateAuditRun(
	ctx context.Context,
	params CreateAuditRunParams,
) (WorkflowRun, error) {
	if err := validateCreateRun(params.CreateRunParams); err != nil {
		return WorkflowRun{}, err
	}
	if params.ProjectID == nil {
		return WorkflowRun{}, invalidf("Audit-managed Run requires Project membership")
	}
	if err := validateOpaque("auditExecutionID", params.AuditExecutionID); err != nil {
		return WorkflowRun{}, err
	}
	if err := validateIdempotencyKey(params.AuditSubmissionKey); err != nil {
		return WorkflowRun{}, invalidf("Audit submission key is invalid")
	}
	metadataLabels, _ := NormalizeRunMetadataLabels(params.MetadataLabels)
	parameters := params.Parameters
	if parameters == nil {
		parameters = map[string]string{}
	}
	encodedParameters, err := json.Marshal(parameters)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("create Audit WorkflowRun: encode parameters: %w", err)
	}
	encodedRuntimeConfig, err := json.Marshal(params.RuntimeConfig)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("create Audit WorkflowRun: encode RuntimeConfig snapshot: %w", err)
	}
	encodedProjectHTTPTarget, err := encodeProjectHTTPTarget(params.ProjectHTTPTarget)
	if err != nil {
		return WorkflowRun{}, err
	}
	encodedMetadataLabels, err := json.Marshal(metadataLabels)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("create Audit WorkflowRun: encode metadata labels: %w", err)
	}
	result, err := scanWorkflowRun(s.db.QueryRow(ctx, `
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
SELECT `+prefixedWorkflowRunColumns("inserted_run")+`
FROM inserted_run`,
		params.RunID, params.OwnerID, params.ProjectID, params.WorkflowName, params.WorkflowVersion,
		params.WorkflowSchemaVersion, []byte(params.WorkflowSnapshot), encodedParameters,
		params.RuntimeConfig.ExplicitLabels(), encodedRuntimeConfig, encodedProjectHTTPTarget,
		params.AuditExecutionID, params.AuditSubmissionKey, encodedMetadataLabels,
	))
	if err != nil {
		switch persistencepostgres.SQLState(err) {
		case "55000":
			return WorkflowRun{}, fmt.Errorf("create Audit WorkflowRun %q: %w", params.RunID, ErrProjectDeleting)
		case "23505":
			return WorkflowRun{}, fmt.Errorf("create Audit WorkflowRun %q: %w", params.RunID, ErrConflict)
		}
		return WorkflowRun{}, fmt.Errorf("create Audit WorkflowRun %q: %w", params.RunID, err)
	}
	result.MetadataLabels = metadataLabels
	return result, nil
}
