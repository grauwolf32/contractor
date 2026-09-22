package artifacts

// Fork and pin statements. Each resolves the source binding and writes the
// target in one statement, and reports back which halves existed so the
// caller can tell a missing source from an idempotent repeat.
// See postgres_fork.go for the callers.

// forkInputSQL copies a source binding revision into a target scope's
// input slot, leaving an existing identical target untouched.
var forkInputSQL = `
WITH source_selection AS (
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
      AND (
          $1 <> 'project'
          OR EXISTS (
              SELECT 1 FROM workflow_runs
              WHERE run_id = $7 AND project_id = $2
          )
      )
      AND (
          ($5::text IS NULL AND revision.revision = binding.current_revision)
          OR ($5::text IS NOT NULL AND revision.revision = $5)
      )
    FOR KEY SHARE OF source_scope
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
        pin_kind, pin_id, scope_kind, scope_id, namespace, name, revision, run_id
    )
    SELECT 'run_input', $7 || ':' || $8, $1, $2, $3, $4, source_selection.revision, $7
    FROM source_selection, lineage
    ON CONFLICT DO NOTHING
    RETURNING 1
)
SELECT
    EXISTS(SELECT 1 FROM source_selection),
    EXISTS(SELECT 1 FROM target_revision),
    (SELECT revision FROM source_selection),
    (SELECT media_type FROM source_selection),
    (SELECT size_bytes FROM source_selection)`

// bindOutputExactSQL binds a resolved source revision to a Run's output
// slot at the expected output revision.
var bindOutputExactSQL = `
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
        pin_kind, pin_id, scope_kind, scope_id, namespace, name, revision, run_id
    )
    SELECT 'run_output', $2 || ':' || $6, $1, $2, $3, $4, $5, $2
    FROM lineage
    ON CONFLICT DO NOTHING
    RETURNING 1
)
SELECT
    EXISTS(SELECT 1 FROM source_selection),
    EXISTS(SELECT 1 FROM target_revision),
    (SELECT media_type FROM source_selection),
    (SELECT size_bytes FROM source_selection)`

// pinExactSQL records a Run's pin on an exact artifact revision and
// reports whether the pin was created or already matched.
var pinExactSQL = `
WITH source_selection AS (
    SELECT 1
    FROM artifact_binding_revisions
    WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4 AND revision = $5
), run_selection AS (
    SELECT 1 FROM workflow_runs WHERE run_id = $8
), inserted AS (
    INSERT INTO artifact_pins (
        pin_kind, pin_id, scope_kind, scope_id, namespace, name, revision, run_id
    )
    SELECT $6, $7, $1, $2, $3, $4, $5, $8
    FROM source_selection, run_selection
    ON CONFLICT DO NOTHING
    RETURNING 1
), existing AS (
    SELECT 1
    FROM artifact_pins
    WHERE pin_kind = $6 AND pin_id = $7
      AND scope_kind = $1 AND scope_id = $2
      AND namespace = $3 AND name = $4 AND revision = $5
      AND run_id = $8
)
SELECT EXISTS(SELECT 1 FROM source_selection),
       EXISTS(SELECT 1 FROM run_selection),
       EXISTS(SELECT 1 FROM inserted) OR EXISTS(SELECT 1 FROM existing)`
