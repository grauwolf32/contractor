package artifacts

import (
	"context"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func (r *PostgresRepository) ForkInput(
	ctx context.Context,
	sourceScope Scope,
	sourceRef ArtifactRef,
	targetScope Scope,
	inputSlot string,
) (ForkResult, error) {
	if err := validateScope(sourceScope); err != nil ||
		(sourceScope.kind != ScopeUser && sourceScope.kind != ScopeProject) {
		return ForkResult{}, ErrInvalidScope
	}
	if err := validateScope(targetScope); err != nil || targetScope.kind != ScopeRun {
		return ForkResult{}, ErrInvalidScope
	}
	if err := validateRef(sourceRef); err != nil {
		return ForkResult{}, err
	}
	if err := validateComponent(inputSlot); err != nil {
		return ForkResult{}, err
	}
	targetRevision, err := r.newID("rev_")
	if err != nil {
		return ForkResult{}, err
	}

	var sourceExists bool
	var targetCreated bool
	var resolvedSourceRevision *string
	var mediaType *string
	var size *int64
	err = r.db.QueryRow(ctx, forkInputSQL,
		sourceScope.kind, sourceScope.id, sourceRef.Namespace, sourceRef.Name, sourceRef.Revision,
		targetScope.kind, targetScope.id, inputSlot, targetRevision,
	).Scan(&sourceExists, &targetCreated, &resolvedSourceRevision, &mediaType, &size)
	if err != nil {
		if persistencepostgres.SQLState(err) == "23503" {
			return ForkResult{}, ErrInvalidScope
		}
		return ForkResult{}, fmt.Errorf("fork Workflow input %q: %w", inputSlot, err)
	}
	if !sourceExists {
		return ForkResult{}, ErrArtifactNotFound
	}
	if !targetCreated {
		target := ArtifactRef{Namespace: "inputs", Name: inputSlot}
		return ForkResult{}, &ConflictError{Ref: target}
	}
	if resolvedSourceRevision == nil || mediaType == nil || size == nil {
		return ForkResult{}, ErrArtifactIntegrity
	}
	resolvedTarget := targetRevision
	return ForkResult{
		SourceRef: ArtifactRef{
			Namespace: sourceRef.Namespace, Name: sourceRef.Name, Revision: resolvedSourceRevision,
		},
		TargetRef: ArtifactRef{Namespace: "inputs", Name: inputSlot, Revision: &resolvedTarget},
		MediaType: *mediaType, Size: *size,
	}, nil
}

func (r *PostgresRepository) BindOutputExact(
	ctx context.Context,
	runScope Scope,
	outputSlot string,
	sourceRef ArtifactRef,
	expectedOutputRevision *string,
) (ForkResult, error) {
	if err := validateScope(runScope); err != nil || runScope.kind != ScopeRun {
		return ForkResult{}, ErrInvalidScope
	}
	if err := validateComponent(outputSlot); err != nil {
		return ForkResult{}, err
	}
	if _, err := exactRevision(sourceRef); err != nil {
		return ForkResult{}, err
	}
	if expectedOutputRevision != nil {
		if err := validateRevision(*expectedOutputRevision); err != nil {
			return ForkResult{}, err
		}
	}
	targetRevision, err := r.newID("rev_")
	if err != nil {
		return ForkResult{}, err
	}

	var sourceExists bool
	var targetCreated bool
	var mediaType *string
	var size *int64
	err = r.db.QueryRow(ctx, bindOutputExactSQL,
		runScope.kind, runScope.id, sourceRef.Namespace, sourceRef.Name, *sourceRef.Revision,
		outputSlot, expectedOutputRevision, targetRevision,
	).Scan(&sourceExists, &targetCreated, &mediaType, &size)
	if err != nil {
		return ForkResult{}, fmt.Errorf("bind Workflow output %q: %w", outputSlot, err)
	}
	if !sourceExists {
		return ForkResult{}, ErrArtifactNotFound
	}
	if !targetCreated {
		target := ArtifactRef{Namespace: "outputs", Name: outputSlot}
		return ForkResult{}, r.writeConflict(ctx, runScope, target, expectedOutputRevision)
	}
	if mediaType == nil || size == nil {
		return ForkResult{}, ErrArtifactIntegrity
	}
	resolvedSource := *sourceRef.Revision
	resolvedTarget := targetRevision
	return ForkResult{
		SourceRef: ArtifactRef{
			Namespace: sourceRef.Namespace, Name: sourceRef.Name, Revision: &resolvedSource,
		},
		TargetRef: ArtifactRef{Namespace: "outputs", Name: outputSlot, Revision: &resolvedTarget},
		MediaType: *mediaType, Size: *size,
	}, nil
}

func (r *PostgresRepository) PinExact(
	ctx context.Context,
	runID string,
	scope Scope,
	ref ArtifactRef,
	kind PinKind,
	pinID string,
) error {
	if _, err := RunScope(runID); err != nil {
		return err
	}
	if err := validateScope(scope); err != nil {
		return err
	}
	if _, err := exactRevision(ref); err != nil {
		return err
	}
	if kind != PinRunInput && kind != PinStageContext && kind != PinStageResult &&
		kind != PinRunOutput && kind != PinFindingProposal && kind != PinFindingEvidence {
		return ErrInvalidName
	}
	if err := validatePinID(pinID); err != nil {
		return ErrInvalidName
	}
	var sourceExists, runExists, matched bool
	err := r.db.QueryRow(ctx, pinExactSQL,
		scope.kind, scope.id, ref.Namespace, ref.Name, *ref.Revision, kind, pinID, runID,
	).Scan(&sourceExists, &runExists, &matched)
	if err != nil {
		return fmt.Errorf("pin exact artifact: %w", err)
	}
	if !sourceExists {
		return ErrArtifactNotFound
	}
	if !runExists {
		return ErrInvalidScope
	}
	if !matched {
		return ErrArtifactIntegrity
	}
	return nil
}

func (r *PostgresRepository) FreezeOutputs(ctx context.Context, scope Scope) error {
	if err := validateScope(scope); err != nil || scope.kind != ScopeRun {
		return ErrInvalidScope
	}
	var runExists bool
	err := r.db.QueryRow(ctx, `
WITH updated AS (
    UPDATE artifact_bindings
    SET frozen = true, updated_at = clock_timestamp()
    WHERE scope_kind = $1 AND scope_id = $2 AND namespace = 'outputs'
    RETURNING 1
)
SELECT EXISTS(SELECT 1 FROM workflow_runs WHERE run_id = $2)`, scope.kind, scope.id).Scan(&runExists)
	if err != nil {
		return fmt.Errorf("freeze Run outputs: %w", err)
	}
	if !runExists {
		return ErrArtifactNotFound
	}
	return nil
}
