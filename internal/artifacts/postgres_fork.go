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
	if err := validateScope(sourceScope); err != nil || sourceScope.kind != ScopeUser {
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
	err = r.db.QueryRow(ctx, `
WITH source_selection AS (
    SELECT revision.revision, revision.version_id, version.media_type, blob.size_bytes
    FROM artifact_binding_revisions AS revision
    JOIN artifact_bindings AS binding
      ON binding.scope_kind = revision.scope_kind
     AND binding.scope_id = revision.scope_id
     AND binding.namespace = revision.namespace
     AND binding.name = revision.name
    JOIN artifact_versions AS version ON version.version_id = revision.version_id
    JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
    WHERE revision.scope_kind = $1 AND revision.scope_id = $2
      AND revision.namespace = $3 AND revision.name = $4
      AND (
          ($5::text IS NULL AND revision.revision = binding.current_revision)
          OR ($5::text IS NOT NULL AND revision.revision = $5)
      )
), inserted_scope AS (
    INSERT INTO artifact_scopes (scope_kind, scope_id)
    SELECT $6, $7 FROM source_selection LIMIT 1
    ON CONFLICT DO NOTHING
    RETURNING 1
), scope_ready AS (
    SELECT 1 FROM inserted_scope
    UNION
    SELECT 1 FROM artifact_scopes WHERE scope_kind = $6 AND scope_id = $7
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision
    )
    SELECT $6, $7, 'inputs', $8, $9
    FROM source_selection, scope_ready
    ON CONFLICT DO NOTHING
    RETURNING 1
), target_revision AS (
    INSERT INTO artifact_binding_revisions (
        scope_kind, scope_id, namespace, name, revision, version_id
    )
    SELECT $6, $7, 'inputs', $8, $9, source_selection.version_id
    FROM source_selection, created_binding
    RETURNING revision
), lineage AS (
    INSERT INTO artifact_lineage (
        target_scope_kind, target_scope_id, target_namespace, target_name, target_revision,
        source_scope_kind, source_scope_id, source_namespace, source_name, source_revision,
        lineage_kind
    )
    SELECT $6, $7, 'inputs', $8, $9,
           $1, $2, $3, $4, source_selection.revision, 'input_fork'
    FROM source_selection, target_revision
    RETURNING 1
), pinned AS (
    INSERT INTO artifact_pins (
        pin_kind, pin_id, scope_kind, scope_id, namespace, name, revision
    )
    SELECT 'run_input', $7 || ':' || $8, $1, $2, $3, $4, source_selection.revision
    FROM source_selection, lineage
    ON CONFLICT DO NOTHING
    RETURNING 1
)
SELECT
    EXISTS(SELECT 1 FROM source_selection),
    EXISTS(SELECT 1 FROM target_revision),
    (SELECT revision FROM source_selection),
    (SELECT media_type FROM source_selection),
    (SELECT size_bytes FROM source_selection)`,
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
	err = r.db.QueryRow(ctx, `
WITH source_selection AS (
    SELECT revision.version_id, version.media_type, blob.size_bytes
    FROM artifact_binding_revisions AS revision
    JOIN artifact_versions AS version ON version.version_id = revision.version_id
    JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
    WHERE revision.scope_kind = $1 AND revision.scope_id = $2
      AND revision.namespace = $3 AND revision.name = $4 AND revision.revision = $5
), updated_binding AS (
    UPDATE artifact_bindings
    SET current_revision = $8, updated_at = clock_timestamp()
    WHERE scope_kind = $1 AND scope_id = $2 AND namespace = 'outputs' AND name = $6
      AND $7::text IS NOT NULL AND current_revision = $7 AND NOT frozen
    RETURNING 1
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision
    )
    SELECT $1, $2, 'outputs', $6, $8 FROM source_selection
    WHERE $7::text IS NULL
    ON CONFLICT DO NOTHING
    RETURNING 1
), claimed_binding AS (
    SELECT 1 FROM updated_binding
    UNION ALL
    SELECT 1 FROM created_binding
), target_revision AS (
    INSERT INTO artifact_binding_revisions (
        scope_kind, scope_id, namespace, name, revision, version_id
    )
    SELECT $1, $2, 'outputs', $6, $8, source_selection.version_id
    FROM source_selection, claimed_binding
    RETURNING revision
), lineage AS (
    INSERT INTO artifact_lineage (
        target_scope_kind, target_scope_id, target_namespace, target_name, target_revision,
        source_scope_kind, source_scope_id, source_namespace, source_name, source_revision,
        lineage_kind
    )
    SELECT $1, $2, 'outputs', $6, $8,
           $1, $2, $3, $4, $5, 'output_bind'
    FROM target_revision
    RETURNING 1
), pinned AS (
    INSERT INTO artifact_pins (
        pin_kind, pin_id, scope_kind, scope_id, namespace, name, revision
    )
    SELECT 'run_output', $2 || ':' || $6, $1, $2, $3, $4, $5
    FROM lineage
    ON CONFLICT DO NOTHING
    RETURNING 1
)
SELECT
    EXISTS(SELECT 1 FROM source_selection),
    EXISTS(SELECT 1 FROM target_revision),
    (SELECT media_type FROM source_selection),
    (SELECT size_bytes FROM source_selection)`,
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
	scope Scope,
	ref ArtifactRef,
	kind PinKind,
	pinID string,
) error {
	if err := validateScope(scope); err != nil {
		return err
	}
	if _, err := exactRevision(ref); err != nil {
		return err
	}
	if kind != PinRunInput && kind != PinStageContext && kind != PinStageResult && kind != PinRunOutput {
		return ErrInvalidName
	}
	if err := validateComponent(pinID); err != nil {
		return ErrInvalidName
	}
	var sourceExists bool
	err := r.db.QueryRow(ctx, `
WITH source_selection AS (
    SELECT 1
    FROM artifact_binding_revisions
    WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4 AND revision = $5
), inserted AS (
    INSERT INTO artifact_pins (
        pin_kind, pin_id, scope_kind, scope_id, namespace, name, revision
    )
    SELECT $6, $7, $1, $2, $3, $4, $5 FROM source_selection
    ON CONFLICT DO NOTHING
    RETURNING 1
)
SELECT EXISTS(SELECT 1 FROM source_selection)`,
		scope.kind, scope.id, ref.Namespace, ref.Name, *ref.Revision, kind, pinID,
	).Scan(&sourceExists)
	if err != nil {
		return fmt.Errorf("pin exact artifact: %w", err)
	}
	if !sourceExists {
		return ErrArtifactNotFound
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
