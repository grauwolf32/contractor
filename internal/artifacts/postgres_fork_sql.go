package artifacts

// SQL statements for postgres_fork.go.

// forkSQL creates binding $8/$9 in scope $6/$7 at revision $10, frozen when
// $15, from the source revision $5 (NULL: the current revision) of binding
// $3/$4 in scope $1/$2. The target reuses the source version without copying
// the blob, and lineage of kind $11 records the exact source revision. $12
// admits only the frozen current revision of the source binding; $13/$14 name
// a Run and the Project it must belong to, or are NULL. The statement holds FOR
// KEY SHARE on the source scope, so a purge cannot remove the source before
// the fork commits.
//
// The target is create-only: an existing binding creates nothing. Beside the
// source-found and target-created flags, the source revision, media type and
// size, it returns the current revision of an existing target and the source
// revision of that revision's lineage of kind $11, for the replay check.
// Used by PostgresRepository.Fork.
var forkSQL = `
WITH source_selection AS MATERIALIZED (
    SELECT revision.revision, revision.version_id, version.media_type, blob.size_bytes
    FROM artifact_binding_revisions AS revision
    JOIN artifact_scopes AS source_scope
      ON source_scope.scope_kind = revision.scope_kind AND source_scope.scope_id = revision.scope_id
    JOIN artifact_bindings AS binding
      ON binding.scope_kind = revision.scope_kind
     AND binding.scope_id = revision.scope_id
     AND binding.namespace = revision.namespace
     AND binding.name = revision.name
    JOIN artifact_versions AS version ON version.version_id = revision.version_id
    JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
    WHERE revision.scope_kind = $1 AND revision.scope_id = $2
      AND revision.namespace = $3 AND revision.name = $4
      AND revision.revision = COALESCE($5::text, binding.current_revision)
      AND (NOT $12::boolean OR (binding.frozen AND binding.current_revision = revision.revision))
      AND ($13::text IS NULL OR EXISTS (
          SELECT 1 FROM workflow_runs AS run
          WHERE run.run_id = $13 AND run.project_id = $14::text
      ))
    FOR KEY SHARE OF source_scope
), existing_target AS (
    SELECT binding.current_revision, lineage.source_revision
    FROM artifact_bindings AS binding
    LEFT JOIN artifact_lineage AS lineage
      ON lineage.target_scope_kind = binding.scope_kind
     AND lineage.target_scope_id = binding.scope_id
     AND lineage.target_namespace = binding.namespace
     AND lineage.target_name = binding.name
     AND lineage.target_revision = binding.current_revision
     AND lineage.lineage_kind = $11
    WHERE binding.scope_kind = $6 AND binding.scope_id = $7
      AND binding.namespace = $8 AND binding.name = $9
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision, frozen
    )
    SELECT $6, $7, $8, $9, $10, $15::boolean FROM source_selection
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
           $1, $2, $3, $4, source_selection.revision, $11
    FROM source_selection, target_revision
    RETURNING 1
)
SELECT
    EXISTS(SELECT 1 FROM source_selection),
    EXISTS(SELECT 1 FROM lineage),
    (SELECT revision FROM source_selection),
    (SELECT media_type FROM source_selection),
    (SELECT size_bytes FROM source_selection),
    (SELECT current_revision FROM existing_target),
    (SELECT source_revision FROM existing_target)`
