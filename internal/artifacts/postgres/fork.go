package postgres

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/jackc/pgx/v5"
)

func (r *Repository) Fork(ctx context.Context, fork artifacts.Fork) (artifacts.ForkResult, error) {
	if fork.ExpectedRevision != nil {
		return r.replaceFork(ctx, fork)
	}
	targetRevision, err := r.newID("rev_")
	if err != nil {
		return artifacts.ForkResult{}, err
	}
	// A source in the target scope proves that the scope exists.
	if fork.Target != fork.Source {
		if err := r.ensureScope(ctx, fork.Target); err != nil {
			return artifacts.ForkResult{}, err
		}
	}
	// A fork between a ProjectScope and a RunScope requires the Run to belong
	// to the Project.
	sourceID, targetID := fork.Source.ID(), fork.Target.ID()
	var memberRun, memberProject *string
	switch {
	case fork.Source.Kind() == artifacts.ScopeProject && fork.Target.Kind() == artifacts.ScopeRun:
		memberRun, memberProject = &targetID, &sourceID
	case fork.Source.Kind() == artifacts.ScopeRun && fork.Target.Kind() == artifacts.ScopeProject:
		memberRun, memberProject = &sourceID, &targetID
	}
	target := artifacts.ArtifactRef{Namespace: fork.TargetRef.Namespace, Name: fork.TargetRef.Name}
	var sourceExists, targetCreated bool
	var sourceRevision, mediaType *string
	var size *int64
	err = r.db.QueryRow(ctx, forkSQL,
		fork.Source.Kind(), fork.Source.ID(), fork.SourceRef.Namespace, fork.SourceRef.Name, fork.SourceRef.Revision,
		fork.Target.Kind(), fork.Target.ID(), target.Namespace, target.Name, targetRevision,
		fork.Kind, fork.SourceFrozen, memberRun, memberProject, fork.Frozen,
	).Scan(&sourceExists, &targetCreated, &sourceRevision, &mediaType, &size)
	if err != nil {
		return artifacts.ForkResult{}, storeError(fmt.Sprintf("fork artifact %s/%s", target.Namespace, target.Name), err,
			&artifacts.ConflictError{Ref: target})
	}
	if !sourceExists {
		return artifacts.ForkResult{}, artifacts.ErrArtifactNotFound
	}
	if sourceRevision == nil || mediaType == nil || size == nil {
		return artifacts.ForkResult{}, blob.ErrIntegrity
	}
	if !targetCreated {
		if !fork.Replay {
			return artifacts.ForkResult{}, &artifacts.ConflictError{Ref: target}
		}
		// The insertion may have waited for a concurrent winner that was not
		// visible to forkSQL's snapshot. Resolve its exact lineage afresh.
		err := r.db.QueryRow(ctx, forkReplaySQL,
			fork.Target.Kind(), fork.Target.ID(), target.Namespace, target.Name,
			fork.Source.Kind(), fork.Source.ID(), fork.SourceRef.Namespace, fork.SourceRef.Name,
			*sourceRevision, fork.Kind,
		).Scan(&targetRevision)
		if errors.Is(err, pgx.ErrNoRows) {
			return artifacts.ForkResult{}, &artifacts.ConflictError{Ref: target}
		}
		if err != nil {
			return artifacts.ForkResult{}, fmt.Errorf("resolve artifact fork replay: %w", err)
		}
	}
	return artifacts.ForkResult{
		SourceRef: artifactExactRef(fork.SourceRef.Namespace, fork.SourceRef.Name, *sourceRevision),
		TargetRef: artifactExactRef(target.Namespace, target.Name, targetRevision),
		MediaType: *mediaType,
		Size:      *size,
	}, nil
}

func (r *Repository) RequireExact(ctx context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef) error {
	var exists bool
	err := r.db.QueryRow(ctx, `
SELECT true FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4 AND revision = $5
FOR KEY SHARE`, scope.Kind(), scope.ID(), ref.Namespace, ref.Name, *ref.Revision).Scan(&exists)
	if errors.Is(err, pgx.ErrNoRows) {
		return artifacts.ErrArtifactNotFound
	}
	if err != nil {
		return fmt.Errorf("require exact artifact: %w", err)
	}
	return nil
}

func (r *Repository) Freeze(ctx context.Context, scope artifacts.Scope, namespace string) error {
	if _, err := r.db.Exec(ctx, `
UPDATE artifact_bindings
SET frozen = true, updated_at = clock_timestamp()
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3`, scope.Kind(), scope.ID(), namespace); err != nil {
		return fmt.Errorf("freeze artifact namespace %s: %w", namespace, err)
	}
	return nil
}

// ThawOutputs reopens the outputs bindings of a RunScope, which FreezeOutputs
// froze when the Run ended, for a continued attempt of that Run. The database
// permits it only in the transaction that records the Run's continuation
// receipt, while the Run is pending and its new attempt preparing; anywhere
// else the thaw fails with a check violation and changes nothing.
func (r *TxRepository) ThawOutputs(ctx context.Context, scope artifacts.Scope) error {
	if err := artifacts.ValidateScope(scope); err != nil || scope.Kind() != artifacts.ScopeRun {
		return artifacts.ErrInvalidScope
	}
	if _, err := r.tx.Exec(ctx, `
UPDATE artifact_bindings
SET frozen = false, updated_at = clock_timestamp()
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = 'outputs'`, scope.Kind(), scope.ID()); err != nil {
		return fmt.Errorf("thaw Run outputs: %w", err)
	}
	return nil
}
