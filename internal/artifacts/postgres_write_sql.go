package artifacts

// SQL statements for postgres_write.go.

// writeSQL stores revision $5 of a mutable artifact binding while holding FOR
// KEY SHARE on its artifact_scopes row. It advances an unfrozen binding whose
// current_revision equals $6 (CAS), or creates the binding when $6 is NULL,
// then upserts the sha256-keyed blob ($14: verified reusable file key), version
// and revision. Returns revision, media type, size, timestamps and the stored
// object key; no row is a conflict. Used by PostgresRepository.Write.
var writeSQL = `
WITH scope_ready AS (
    SELECT 1 FROM artifact_scopes WHERE scope_kind = $1 AND scope_id = $2 FOR KEY SHARE
), updated_binding AS (
    UPDATE artifact_bindings
    SET current_revision = $5, updated_at = clock_timestamp()
    WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4
      AND $6::text IS NOT NULL AND current_revision = $6 AND NOT frozen
      AND EXISTS (SELECT 1 FROM scope_ready)
    RETURNING created_at
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision
    )
    SELECT $1, $2, $3, $4, $5 FROM scope_ready
    WHERE $6::text IS NULL
    ON CONFLICT DO NOTHING
    RETURNING created_at
), claimed_binding AS (
    SELECT created_at FROM updated_binding
    UNION ALL
    SELECT created_at FROM created_binding
), inserted_blob AS (
    INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend, object_key)
    SELECT $8, $9, $10, $12, $13 FROM claimed_binding
    ON CONFLICT (sha256) DO UPDATE
    SET sha256 = EXCLUDED.sha256,
        object_key = CASE
            WHEN artifact_blobs.backend = 'filesystem'
             AND artifact_blobs.object_key IS DISTINCT FROM $14::text
            THEN EXCLUDED.object_key ELSE artifact_blobs.object_key END
    WHERE artifact_blobs.backend = EXCLUDED.backend
      AND (artifact_blobs.backend = 'filesystem' OR artifact_blobs.payload = EXCLUDED.payload)
      AND artifact_blobs.size_bytes = EXCLUDED.size_bytes
    RETURNING object_key
), blob_ready AS (
    SELECT 1 FROM inserted_blob
), inserted_version AS (
    INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
    SELECT $7, $8, $11 FROM claimed_binding, blob_ready
    RETURNING version_id
), inserted_revision AS (
    INSERT INTO artifact_binding_revisions (
        scope_kind, scope_id, namespace, name, revision, version_id
    )
    SELECT $1, $2, $3, $4, $5, version_id FROM inserted_version
    RETURNING revision, created_at
)
SELECT inserted_revision.revision, $11::text, $10::bigint,
       claimed_binding.created_at, inserted_revision.created_at, (SELECT object_key FROM inserted_blob)
FROM inserted_revision, claimed_binding`
