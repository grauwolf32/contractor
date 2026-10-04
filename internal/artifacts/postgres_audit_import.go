package artifacts

import (
	"context"
	"fmt"
	"strings"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// ImportAuditArtifact retains one exact Run revision in the owning Project.
// Source selection and target/lineage creation are one SQL statement, while
// the trusted caller makes the deterministic target externally visible only
// by committing its Audit-owned receipt. The caller, not publication_mode,
// proves whether this is an Audit child result or an ordinary finding import.
func (r *PostgresRepository) ImportAuditArtifact(
	ctx context.Context,
	runScope Scope,
	sourceRef ArtifactRef,
	projectScope Scope,
	targetRef ArtifactRef,
) (ForkResult, error) {
	if err := validateScope(runScope); err != nil || runScope.kind != ScopeRun {
		return ForkResult{}, ErrInvalidScope
	}
	if err := validateScope(projectScope); err != nil || projectScope.kind != ScopeProject {
		return ForkResult{}, ErrInvalidScope
	}
	if _, err := exactRevision(sourceRef); err != nil {
		return ForkResult{}, err
	}
	if targetRef.Revision != nil || !strings.HasPrefix(targetRef.Namespace, "audit-") {
		return ForkResult{}, ErrInvalidName
	}
	if err := validateRef(targetRef); err != nil {
		return ForkResult{}, err
	}
	targetRevision, err := r.newID("rev_")
	if err != nil {
		return ForkResult{}, err
	}

	if _, err := r.db.Exec(ctx, `
INSERT INTO artifact_scopes (scope_kind, scope_id)
VALUES ($1, $2)
ON CONFLICT DO NOTHING`, projectScope.kind, projectScope.id); err != nil {
		switch persistencepostgres.SQLState(err) {
		case "55000":
			return ForkResult{}, fmt.Errorf("create Audit artifact scope: %w", ErrScopeDeleting)
		case persistencepostgres.SQLStateForeignKeyViolation:
			return ForkResult{}, fmt.Errorf("create Audit artifact scope: %w", ErrInvalidScope)
		default:
			return ForkResult{}, fmt.Errorf("create Audit artifact scope: %w", err)
		}
	}

	var sourceExists, targetCreated bool
	var mediaType *string
	var size *int64
	err = r.db.QueryRow(ctx, importAuditArtifactSQL,
		runScope.kind, runScope.id, sourceRef.Namespace, sourceRef.Name, *sourceRef.Revision,
		projectScope.kind, projectScope.id, targetRef.Namespace, targetRef.Name, targetRevision,
	).Scan(&sourceExists, &targetCreated, &mediaType, &size)
	if err != nil {
		switch persistencepostgres.SQLState(err) {
		case "55000":
			return ForkResult{}, fmt.Errorf("import Audit artifact: %w", ErrScopeDeleting)
		case persistencepostgres.SQLStateForeignKeyViolation:
			return ForkResult{}, fmt.Errorf("import Audit artifact: %w", ErrInvalidScope)
		case persistencepostgres.SQLStateUniqueViolation:
			return ForkResult{}, &ConflictError{Ref: targetRef}
		default:
			return ForkResult{}, fmt.Errorf("import Audit artifact: %w", err)
		}
	}
	if !sourceExists {
		return ForkResult{}, ErrArtifactNotFound
	}
	if !targetCreated {
		return ForkResult{}, &ConflictError{Ref: targetRef}
	}
	if mediaType == nil || size == nil {
		return ForkResult{}, ErrArtifactIntegrity
	}
	sourceRevision := *sourceRef.Revision
	return ForkResult{
		SourceRef: ArtifactRef{Namespace: sourceRef.Namespace, Name: sourceRef.Name, Revision: &sourceRevision},
		TargetRef: ArtifactRef{Namespace: targetRef.Namespace, Name: targetRef.Name, Revision: &targetRevision},
		MediaType: *mediaType, Size: *size,
	}, nil
}
