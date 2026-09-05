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
		if persistencepostgres.SQLState(err) == "23503" {
			return ForkResult{}, fmt.Errorf("create Project artifact scope: %w", ErrInvalidScope)
		}
		return ForkResult{}, fmt.Errorf("create Project artifact scope: %w", err)
	}

	var sourceExists bool
	var targetCreated bool
	var mediaType *string
	var size *int64
	err = r.db.QueryRow(ctx, `
WITH source_selection AS (
    SELECT revision.version_id, version.media_type, blob.size_bytes
    FROM artifact_bindings AS binding
    JOIN artifact_binding_revisions AS revision
      ON revision.scope_kind = binding.scope_kind
     AND revision.scope_id = binding.scope_id
     AND revision.namespace = binding.namespace
     AND revision.name = binding.name
     AND revision.revision = $5
    JOIN artifact_versions AS version ON version.version_id = revision.version_id
    JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
    JOIN workflow_runs AS run ON run.run_id = $2 AND run.project_id = $7
    WHERE binding.scope_kind = $1 AND binding.scope_id = $2
      AND binding.namespace = $3 AND binding.name = $4
      AND binding.current_revision = $5 AND binding.frozen
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision
    )
    SELECT $6, $7, 'outputs', $8, $9 FROM source_selection
    ON CONFLICT DO NOTHING
    RETURNING 1
), target_revision AS (
    INSERT INTO artifact_binding_revisions (
        scope_kind, scope_id, namespace, name, revision, version_id
    )
    SELECT $6, $7, 'outputs', $8, $9, source_selection.version_id
    FROM source_selection, created_binding
    RETURNING revision
), lineage AS (
    INSERT INTO artifact_lineage (
        target_scope_kind, target_scope_id, target_namespace, target_name, target_revision,
        source_scope_kind, source_scope_id, source_namespace, source_name, source_revision,
        lineage_kind
    )
    SELECT $6, $7, 'outputs', $8, $9,
           $1, $2, $3, $4, $5, 'project_output_publish'
    FROM target_revision
    RETURNING 1
)
SELECT
    EXISTS(SELECT 1 FROM source_selection),
    EXISTS(SELECT 1 FROM lineage),
    (SELECT media_type FROM source_selection),
    (SELECT size_bytes FROM source_selection)`,
		runScope.kind, runScope.id, sourceRef.Namespace, sourceRef.Name, *sourceRef.Revision,
		projectScope.kind, projectScope.id, outputSlot, targetRevision,
	).Scan(&sourceExists, &targetCreated, &mediaType, &size)
	if err != nil {
		switch persistencepostgres.SQLState(err) {
		case "55000":
			return ForkResult{}, fmt.Errorf("publish Project output %q: %w", outputSlot, ErrScopeDeleting)
		case "23503":
			return ForkResult{}, fmt.Errorf("publish Project output %q: %w", outputSlot, ErrInvalidScope)
		case "23505":
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
