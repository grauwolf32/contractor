package runstore

import (
	"context"
	"fmt"
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
	run, inserted, err := s.insertRun(ctx, "create Audit WorkflowRun", runRecord{
		CreateRunParams:  params.CreateRunParams,
		auditExecutionID: &params.AuditExecutionID, auditSubmissionKey: &params.AuditSubmissionKey,
	})
	if err != nil {
		return WorkflowRun{}, err
	}
	if !inserted {
		return WorkflowRun{}, fmt.Errorf("create Audit WorkflowRun %q: %w", params.RunID, ErrConflict)
	}
	return run, nil
}
