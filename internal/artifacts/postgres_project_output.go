package artifacts

import (
	"context"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// PublishRunOutput creates a Project output binding without copying payload
// bytes. Callers that need atomicity with a WorkflowRun transition pass a
// transaction-bound repository.
func (r *PostgresRepository) PublishRunOutput(
	ctx context.Context,
	runScope Scope,
	sourceRef ArtifactRef,
	projectScope Scope,
	outputSlot string,
) (ForkResult, error) {
	if err := validateScope(runScope); err != nil || runScope.kind != ScopeRun {
		return ForkResult{}, ErrInvalidScope
	}
	if err := validateScope(projectScope); err != nil || projectScope.kind != ScopeProject {
		return ForkResult{}, ErrInvalidScope
	}
	if err := validateComponent(outputSlot); err != nil {
		return ForkResult{}, err
	}
	if sourceRef.Namespace != "outputs" || sourceRef.Name != outputSlot {
		return ForkResult{}, ErrInvalidName
	}
	if _, err := exactRevision(sourceRef); err != nil {
		return ForkResult{}, err
	}
	targetRevision, err := r.newID("rev_")
	if err != nil {
		return ForkResult{}, err
	}

	// Scope creation is a separate statement for the same concurrency reason as
	// Write: after an INSERT ... DO NOTHING waits for another transaction, the
	// following READ COMMITTED statement sees the winning Project scope.
	if _, err := r.db.Exec(ctx, `
INSERT INTO artifact_scopes (scope_kind, scope_id)
VALUES ($1, $2)
ON CONFLICT DO NOTHING`, projectScope.kind, projectScope.id); err != nil {
		if persistencepostgres.SQLState(err) == "55000" {
			return ForkResult{}, fmt.Errorf("create Project artifact scope: %w", ErrScopeDeleting)
		}
		if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateForeignKeyViolation {
			return ForkResult{}, fmt.Errorf("create Project artifact scope: %w", ErrInvalidScope)
		}
		return ForkResult{}, fmt.Errorf("create Project artifact scope: %w", err)
	}

	var sourceExists bool
	var targetCreated bool
	var mediaType *string
	var size *int64
	err = r.db.QueryRow(ctx, publishRunOutputSQL,
		runScope.kind, runScope.id, sourceRef.Namespace, sourceRef.Name, *sourceRef.Revision,
		projectScope.kind, projectScope.id, outputSlot, targetRevision,
	).Scan(&sourceExists, &targetCreated, &mediaType, &size)
	if err != nil {
		switch persistencepostgres.SQLState(err) {
		case "55000":
			return ForkResult{}, fmt.Errorf("publish Project output %q: %w", outputSlot, ErrScopeDeleting)
		case persistencepostgres.SQLStateForeignKeyViolation:
			return ForkResult{}, fmt.Errorf("publish Project output %q: %w", outputSlot, ErrInvalidScope)
		case persistencepostgres.SQLStateUniqueViolation:
			return ForkResult{}, &ConflictError{
				Ref: ArtifactRef{Namespace: "outputs", Name: outputSlot},
			}
		}
		return ForkResult{}, fmt.Errorf("publish Project output %q: %w", outputSlot, err)
	}
	if !sourceExists {
		return ForkResult{}, ErrArtifactNotFound
	}
	if !targetCreated {
		return ForkResult{}, &ConflictError{
			Ref: ArtifactRef{Namespace: "outputs", Name: outputSlot},
		}
	}
	if mediaType == nil || size == nil {
		return ForkResult{}, ErrArtifactIntegrity
	}
	sourceRevision := *sourceRef.Revision
	return ForkResult{
		SourceRef: ArtifactRef{
			Namespace: sourceRef.Namespace, Name: sourceRef.Name, Revision: &sourceRevision,
		},
		TargetRef: ArtifactRef{
			Namespace: "outputs", Name: outputSlot, Revision: &targetRevision,
		},
		MediaType: *mediaType,
		Size:      *size,
	}, nil
}
