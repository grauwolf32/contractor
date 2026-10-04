package artifacts

// SQL statements for postgres_skill_fork.go.

// forkSkillSQL forks exact skill revision $4 of user scope $1/$2 into a new
// skills/$3 binding of run scope $5/$6 at revision $7, reusing the version,
// and recording 'input_fork' lineage. Beside
// flags, media type and size it returns the existing target revision and its
// input_fork source, so the caller can accept an idempotent retry.
// Used by PostgresRepository.ForkSkill.
var forkSkillSQL = `
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
)
SELECT
    EXISTS(SELECT 1 FROM source_selection),
    EXISTS(SELECT 1 FROM target_revision),
    (SELECT revision FROM source_selection),
    (SELECT current_revision FROM existing_target),
    (SELECT source_revision FROM existing_target),
    (SELECT media_type FROM source_selection),
    (SELECT size_bytes FROM source_selection)`
