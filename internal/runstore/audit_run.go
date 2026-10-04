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
	result, err := scanWorkflowRun(s.db.QueryRow(ctx, createAuditRunSQL+prefixedWorkflowRunColumns("inserted_run")+`
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
		case persistencepostgres.SQLStateUniqueViolation:
			return WorkflowRun{}, fmt.Errorf("create Audit WorkflowRun %q: %w", params.RunID, ErrConflict)
		}
		return WorkflowRun{}, fmt.Errorf("create Audit WorkflowRun %q: %w", params.RunID, err)
	}
	result.MetadataLabels = metadataLabels
	return result, nil
}
