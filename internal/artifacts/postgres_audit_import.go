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
		case "23503":
			return ForkResult{}, fmt.Errorf("create Audit artifact scope: %w", ErrInvalidScope)
		default:
			return ForkResult{}, fmt.Errorf("create Audit artifact scope: %w", err)
		}
	}

	var sourceExists, targetCreated bool
	var mediaType *string
	var size *int64
	err = r.db.QueryRow(ctx, `
WITH source_selection AS MATERIALIZED (
    SELECT revision.version_id, version.media_type, blob.size_bytes
      FROM artifact_binding_revisions AS revision
      JOIN artifact_versions AS version ON version.version_id = revision.version_id
      JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
      JOIN workflow_runs AS run
        ON run.run_id = $2 AND run.project_id = $7
     WHERE revision.scope_kind = $1 AND revision.scope_id = $2
       AND revision.namespace = $3 AND revision.name = $4
       AND revision.revision = $5
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision, frozen
    )
    SELECT $6, $7, $8, $9, $10, true FROM source_selection
    ON CONFLICT DO NOTHING
    RETURNING 1
), target_revision AS (
    INSERT INTO artifact_binding_revisions (
        scope_kind, scope_id, namespace, name, revision, version_id
    )
    SELECT $6, $7, $8, $9, $10, source_selection.version_id
      FROM source_selection, created_binding
    RETURNING revision
), lineage AS (
    INSERT INTO artifact_lineage (
        target_scope_kind, target_scope_id, target_namespace, target_name, target_revision,
        source_scope_kind, source_scope_id, source_namespace, source_name, source_revision,
        lineage_kind
    )
    SELECT $6, $7, $8, $9, $10,
           $1, $2, $3, $4, $5, 'audit_import'
      FROM target_revision
    RETURNING 1
)
SELECT EXISTS(SELECT 1 FROM source_selection),
       EXISTS(SELECT 1 FROM lineage),
       (SELECT media_type FROM source_selection),
       (SELECT size_bytes FROM source_selection)`,
		runScope.kind, runScope.id, sourceRef.Namespace, sourceRef.Name, *sourceRef.Revision,
		projectScope.kind, projectScope.id, targetRef.Namespace, targetRef.Name, targetRevision,
	).Scan(&sourceExists, &targetCreated, &mediaType, &size)
	if err != nil {
		switch persistencepostgres.SQLState(err) {
		case "55000":
			return ForkResult{}, fmt.Errorf("import Audit artifact: %w", ErrScopeDeleting)
		case "23503":
			return ForkResult{}, fmt.Errorf("import Audit artifact: %w", ErrInvalidScope)
		case "23505":
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
