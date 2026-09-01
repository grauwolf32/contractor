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
		if persistencepostgres.SQLState(err) == "23503" {
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
	err = r.db.QueryRow(ctx, `
WITH source_selection AS (
    SELECT revision.revision, revision.version_id, version.media_type, blob.size_bytes
    FROM artifact_binding_revisions AS revision
    JOIN artifact_versions AS version ON version.version_id = revision.version_id
    JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
    WHERE revision.scope_kind = $1 AND revision.scope_id = $2
      AND revision.namespace = 'skills' AND revision.name = $3
      AND revision.revision = $4
), existing_target AS (
    SELECT binding.current_revision, lineage.source_revision
    FROM artifact_bindings AS binding
    LEFT JOIN artifact_lineage AS lineage
      ON lineage.target_scope_kind = binding.scope_kind
     AND lineage.target_scope_id = binding.scope_id
     AND lineage.target_namespace = binding.namespace
     AND lineage.target_name = binding.name
     AND lineage.target_revision = binding.current_revision
     AND lineage.lineage_kind = 'input_fork'
    WHERE binding.scope_kind = $5 AND binding.scope_id = $6
      AND binding.namespace = 'skills' AND binding.name = $3
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision
    )
    SELECT $5, $6, 'skills', $3, $7 FROM source_selection
    ON CONFLICT DO NOTHING
    RETURNING 1
), target_revision AS (
    INSERT INTO artifact_binding_revisions (
        scope_kind, scope_id, namespace, name, revision, version_id
    )
    SELECT $5, $6, 'skills', $3, $7, source_selection.version_id
    FROM source_selection, created_binding
    RETURNING revision
), lineage AS (
    INSERT INTO artifact_lineage (
        target_scope_kind, target_scope_id, target_namespace, target_name, target_revision,
        source_scope_kind, source_scope_id, source_namespace, source_name, source_revision,
        lineage_kind
    )
    SELECT $5, $6, 'skills', $3, $7,
           $1, $2, 'skills', $3, source_selection.revision, 'input_fork'
    FROM source_selection, target_revision
    RETURNING 1
), pinned AS (
    INSERT INTO artifact_pins (
        pin_kind, pin_id, scope_kind, scope_id, namespace, name, revision
    )
    SELECT 'run_input', $6 || ':skill:' || $3,
           $1, $2, 'skills', $3, source_selection.revision
    FROM source_selection, lineage
    ON CONFLICT DO NOTHING
    RETURNING 1
)
SELECT
    EXISTS(SELECT 1 FROM source_selection),
    EXISTS(SELECT 1 FROM target_revision),
    (SELECT revision FROM source_selection),
    (SELECT current_revision FROM existing_target),
    (SELECT source_revision FROM existing_target),
    (SELECT media_type FROM source_selection),
    (SELECT size_bytes FROM source_selection)`,
		sourceScope.kind, sourceScope.id, name, sourceRevision,
		targetScope.kind, targetScope.id, targetRevision,
	).Scan(
		&sourceExists, &targetCreated, &selectedSourceRevision,
		&existingTargetRevision, &existingSourceRevision, &mediaType, &size,
	)
	if err != nil {
		if persistencepostgres.SQLState(err) == "23503" {
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
