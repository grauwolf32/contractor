package artifacts

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

func (r *PostgresRepository) Fork(ctx context.Context, fork Fork) (ForkResult, error) {
	targetRevision, err := r.newID("rev_")
	if err != nil {
		return ForkResult{}, err
	}
	// A source in the target scope proves that the scope exists.
	if fork.Target != fork.Source {
		if err := r.ensureScope(ctx, fork.Target); err != nil {
			return ForkResult{}, err
		}
	}
	// A fork between a ProjectScope and a RunScope requires the Run to belong
	// to the Project.
	var memberRun, memberProject *string
	switch {
	case fork.Source.kind == ScopeProject && fork.Target.kind == ScopeRun:
		memberRun, memberProject = &fork.Target.id, &fork.Source.id
	case fork.Source.kind == ScopeRun && fork.Target.kind == ScopeProject:
		memberRun, memberProject = &fork.Source.id, &fork.Target.id
	}
	target := ArtifactRef{Namespace: fork.TargetRef.Namespace, Name: fork.TargetRef.Name}
	var sourceExists, targetCreated bool
	var sourceRevision, mediaType, existingTarget, existingSource *string
	var size *int64
	err = r.db.QueryRow(ctx, forkSQL,
		fork.Source.kind, fork.Source.id, fork.SourceRef.Namespace, fork.SourceRef.Name, fork.SourceRef.Revision,
		fork.Target.kind, fork.Target.id, target.Namespace, target.Name, targetRevision,
		fork.Kind, fork.SourceFrozen, memberRun, memberProject, fork.Frozen,
	).Scan(&sourceExists, &targetCreated, &sourceRevision, &mediaType, &size, &existingTarget, &existingSource)
	if err != nil {
		return ForkResult{}, storeError(fmt.Sprintf("fork artifact %s/%s", target.Namespace, target.Name), err,
			&ConflictError{Ref: target})
	}
	if !sourceExists {
		return ForkResult{}, ErrArtifactNotFound
	}
	if sourceRevision == nil || mediaType == nil || size == nil {
		return ForkResult{}, ErrArtifactIntegrity
	}
	if !targetCreated {
		if !fork.Replay || existingTarget == nil || existingSource == nil || *existingSource != *sourceRevision {
			return ForkResult{}, &ConflictError{Ref: target}
		}
		targetRevision = *existingTarget
	}
	return ForkResult{
		SourceRef: artifactExactRef(fork.SourceRef.Namespace, fork.SourceRef.Name, *sourceRevision),
		TargetRef: artifactExactRef(target.Namespace, target.Name, targetRevision),
		MediaType: *mediaType,
		Size:      *size,
	}, nil
}

func (r *PostgresRepository) RequireExact(ctx context.Context, scope Scope, ref ArtifactRef) error {
	var exists bool
	err := r.db.QueryRow(ctx, `
SELECT true FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4 AND revision = $5
FOR KEY SHARE`, scope.kind, scope.id, ref.Namespace, ref.Name, *ref.Revision).Scan(&exists)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrArtifactNotFound
	}
	if err != nil {
		return fmt.Errorf("require exact artifact: %w", err)
	}
	return nil
}

func (r *PostgresRepository) Freeze(ctx context.Context, scope Scope, namespace string) error {
	if _, err := r.db.Exec(ctx, `
UPDATE artifact_bindings
SET frozen = true, updated_at = clock_timestamp()
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3`, scope.kind, scope.id, namespace); err != nil {
		return fmt.Errorf("freeze artifact namespace %s: %w", namespace, err)
	}
	return nil
}

// ThawOutputs reopens the outputs bindings of a RunScope, which FreezeOutputs
// froze when the Run ended, for a continued attempt of that Run. The database
// permits it only in the transaction that records the Run's continuation
// receipt, while the Run is pending and its new attempt preparing; anywhere
// else the thaw fails with a check violation and changes nothing.
func (r *PostgresTxRepository) ThawOutputs(ctx context.Context, scope Scope) error {
	if err := validateScope(scope); err != nil || scope.kind != ScopeRun {
		return ErrInvalidScope
	}
	if _, err := r.tx.Exec(ctx, `
UPDATE artifact_bindings
SET frozen = false, updated_at = clock_timestamp()
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = 'outputs'`, scope.kind, scope.id); err != nil {
		return fmt.Errorf("thaw Run outputs: %w", err)
	}
	return nil
}
