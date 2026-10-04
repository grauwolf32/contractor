package artifacts

// SQL statements for postgres_audit_import.go.

// importAuditArtifactSQL retains exact run revision $1..$5 as a new frozen
// project binding $6..$10 that reuses the source version without copying the
// blob, and records 'audit_import' lineage. The source is admitted only when
// run $2 belongs to project $7. Returns source-found and target-created flags,
// media type and size. Used by PostgresRepository.ImportAuditArtifact.
var importAuditArtifactSQL = `
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
       (SELECT size_bytes FROM source_selection)`
