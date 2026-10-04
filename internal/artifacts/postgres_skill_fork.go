package artifacts

import (
	"context"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// ForkSkill reuses the exact owner version in a reserved RunScope binding. A
// retry returns the existing target only when its ordinary lineage points to
// the same exact source.
func (r *PostgresRepository) ForkSkill(
	ctx context.Context,
	sourceScope Scope,
	sourceRef ArtifactRef,
	targetScope Scope,
	name string,
) (ForkResult, error) {
	if err := validateScope(sourceScope); err != nil || sourceScope.kind != ScopeUser {
		return ForkResult{}, ErrInvalidScope
	}
	if err := validateScope(targetScope); err != nil || targetScope.kind != ScopeRun {
		return ForkResult{}, ErrInvalidScope
	}
	if sourceRef.Namespace != "skills" || sourceRef.Name != name {
		return ForkResult{}, ErrInvalidName
	}
	sourceRevision, err := exactRevision(sourceRef)
	if err != nil {
		return ForkResult{}, err
	}
	if err := validateComponent(name); err != nil {
		return ForkResult{}, err
	}
	if _, err := r.db.Exec(ctx, `
INSERT INTO artifact_scopes (scope_kind, scope_id)
VALUES ($1, $2)
ON CONFLICT DO NOTHING`, targetScope.kind, targetScope.id); err != nil {
		if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateForeignKeyViolation {
			return ForkResult{}, ErrInvalidScope
		}
		return ForkResult{}, fmt.Errorf("create Run Skill scope: %w", err)
	}
	targetRevision, err := r.newID("rev_")
	if err != nil {
		return ForkResult{}, err
	}
	var sourceExists, targetCreated bool
	var selectedSourceRevision, existingTargetRevision, existingSourceRevision *string
	var mediaType *string
	var size *int64
	err = r.db.QueryRow(ctx, forkSkillSQL,
		sourceScope.kind, sourceScope.id, name, sourceRevision,
		targetScope.kind, targetScope.id, targetRevision,
	).Scan(
		&sourceExists, &targetCreated, &selectedSourceRevision,
		&existingTargetRevision, &existingSourceRevision, &mediaType, &size,
	)
	if err != nil {
		if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateForeignKeyViolation {
			return ForkResult{}, ErrInvalidScope
		}
		return ForkResult{}, fmt.Errorf("fork Run Skill %q: %w", name, err)
	}
	if !sourceExists || selectedSourceRevision == nil || mediaType == nil || size == nil {
		return ForkResult{}, ErrArtifactNotFound
	}
	resolvedTarget := ""
	if targetCreated {
		resolvedTarget = targetRevision
	} else if existingTargetRevision != nil && existingSourceRevision != nil && *existingSourceRevision == sourceRevision {
		resolvedTarget = *existingTargetRevision
	} else {
		return ForkResult{}, &ConflictError{Ref: ArtifactRef{Namespace: "skills", Name: name}}
	}
	resolvedSource := *selectedSourceRevision
	return ForkResult{
		SourceRef: ArtifactRef{Namespace: "skills", Name: name, Revision: &resolvedSource},
		TargetRef: ArtifactRef{Namespace: "skills", Name: name, Revision: &resolvedTarget},
		MediaType: *mediaType,
		Size:      *size,
	}, nil
}
