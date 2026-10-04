package artifacts

// SQL statements for postgres_project_output.go.

// publishRunOutputSQL publishes a run's frozen outputs binding at current
// revision $5 as project binding outputs/$8 at new revision $9, reusing the
// version and recording 'project_output_publish' lineage. It requires run $2
// to belong to project $7 and holds FOR KEY SHARE on the source scope. Returns
// source-found and target-created flags, media type and size.
// Used by PostgresRepository.PublishRunOutput.
var publishRunOutputSQL = `
WITH source_selection AS (
    SELECT revision.version_id, version.media_type, blob.size_bytes
    FROM artifact_bindings AS binding
    JOIN artifact_scopes AS source_scope
      ON source_scope.scope_kind = binding.scope_kind AND source_scope.scope_id = binding.scope_id
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
    FOR KEY SHARE OF source_scope
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
    (SELECT size_bytes FROM source_selection)`
