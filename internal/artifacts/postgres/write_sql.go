package postgres

// SQL statements for postgres_write.go.

// writeSQL stores revision $5 of an artifact binding while holding FOR KEY
// SHARE on its artifact_scopes row. It advances an unfrozen binding whose
// current_revision equals $6 (CAS), or creates the binding, frozen when $15,
// when $6 is NULL, then upserts the sha256-keyed blob ($14: verified reusable
// file key), version and revision. Returns revision, media type, size,
// timestamps and the stored object key; no row is a conflict. The binding's
// retained-history counters (ART-R123) are charged the new revision's full
// encoded size $10 and credited the $18 revisions and $19 bytes that the
// same transaction removes; an advance whose result would retain more than
// $16 revisions or $17 bytes stores nothing and returns no row, so history
// admission decides it. Used by writeStatement.exec.
var writeSQL = `
WITH scope_ready AS (
    SELECT 1 FROM artifact_scopes WHERE scope_kind = $1 AND scope_id = $2 FOR KEY SHARE
), updated_binding AS (
    UPDATE artifact_bindings
    SET current_revision = $5, updated_at = clock_timestamp(),
        retained_revisions = retained_revisions + 1 - $18::bigint,
        retained_bytes = retained_bytes + $10 - $19::bigint
    WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4
      AND $6::text IS NOT NULL AND current_revision = $6 AND NOT frozen
      AND retained_revisions + 1 - $18::bigint <= $16::bigint
      AND retained_bytes + $10 - $19::bigint <= $17::bigint
      AND EXISTS (SELECT 1 FROM scope_ready)
    RETURNING created_at
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision, frozen,
        retained_revisions, retained_bytes
    )
    SELECT $1, $2, $3, $4, $5, $15::boolean, 1, $10 FROM scope_ready
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
             AND EXCLUDED.backend = 'filesystem'
             AND artifact_blobs.object_key IS DISTINCT FROM $14::text
            THEN EXCLUDED.object_key ELSE artifact_blobs.object_key END
    WHERE ((artifact_blobs.backend = 'postgresql'
            AND (EXCLUDED.backend <> 'postgresql' OR artifact_blobs.payload = EXCLUDED.payload))
        OR (artifact_blobs.backend = 'filesystem'
            AND (EXCLUDED.backend = 'filesystem' OR artifact_blobs.object_key = $14::text)))
      AND artifact_blobs.size_bytes = EXCLUDED.size_bytes
    RETURNING object_key
), blob_ready AS (
    SELECT 1 FROM inserted_blob
), inserted_version AS (
    INSERT INTO artifact_versions (version_id, blob_sha256, media_type, execution_origin)
    SELECT $7, $8, $11, $20::jsonb FROM claimed_binding, blob_ready
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
