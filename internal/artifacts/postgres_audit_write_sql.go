package artifacts

// SQL statements for postgres_audit_write.go.

// writeAuditArtifactSQL creates a frozen binding with its first revision $5 in
// an existing scope. There is no update path: an existing binding yields no
// row (a conflict). Blob, version and revision are stored as in writeSQL ($13:
// verified reusable file key). Returns revision, media type, size, timestamps
// and the stored object key. Used by PostgresRepository.WriteAuditArtifact.
var writeAuditArtifactSQL = `
WITH scope_ready AS (
    SELECT 1 FROM artifact_scopes WHERE scope_kind = $1 AND scope_id = $2
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision, frozen
    )
    SELECT $1, $2, $3, $4, $5, true FROM scope_ready
    ON CONFLICT DO NOTHING
    RETURNING created_at
), inserted_blob AS (
    INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend, object_key)
    SELECT $7, $8, $9, $11, $12 FROM created_binding
    ON CONFLICT (sha256) DO UPDATE
    SET sha256 = EXCLUDED.sha256,
        object_key = CASE
            WHEN artifact_blobs.backend = 'filesystem'
             AND artifact_blobs.object_key IS DISTINCT FROM $13::text
            THEN EXCLUDED.object_key ELSE artifact_blobs.object_key END
    WHERE artifact_blobs.backend = EXCLUDED.backend
      AND (artifact_blobs.backend = 'filesystem' OR artifact_blobs.payload = EXCLUDED.payload)
      AND artifact_blobs.size_bytes = EXCLUDED.size_bytes
    RETURNING object_key
), inserted_version AS (
    INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
    SELECT $6, $7, $10 FROM created_binding, inserted_blob
    RETURNING version_id
), inserted_revision AS (
    INSERT INTO artifact_binding_revisions (
        scope_kind, scope_id, namespace, name, revision, version_id
    )
    SELECT $1, $2, $3, $4, $5, version_id FROM inserted_version
    RETURNING revision, created_at
)
SELECT inserted_revision.revision, $10::text, $9::bigint,
       created_binding.created_at, inserted_revision.created_at, (SELECT object_key FROM inserted_blob)
FROM inserted_revision, created_binding`
