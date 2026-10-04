package artifacts

// SQL statements for prefix_read_batch.go.

// readPrefixBatchSQL reads up to $5 bindings of one scope and namespace whose
// name starts with $4, by name, with their current revision, version and blob.
// LEFT JOINs keep a binding with a dangling revision as NULL columns. Inline
// payloads are returned only when the combined size is within $6, and that
// total is returned on every row. Used by PostgresRepository.ReadPrefixBatch.
var readPrefixBatchSQL = `
WITH limited AS MATERIALIZED (
    SELECT scope_kind, scope_id, namespace, name, current_revision, created_at
    FROM artifact_bindings
    WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3
      AND starts_with(name, $4)
    ORDER BY name
    LIMIT $5
)
SELECT binding.namespace, binding.name, revision.revision, version.media_type,
       blob.backend, blob.object_key,
       CASE WHEN sum(blob.size_bytes) OVER () <= $6 THEN blob.payload END,
       blob.sha256, blob.size_bytes, binding.created_at, revision.created_at,
       sum(blob.size_bytes) OVER ()
FROM limited AS binding
LEFT JOIN artifact_binding_revisions AS revision
  ON revision.scope_kind = binding.scope_kind AND revision.scope_id = binding.scope_id
 AND revision.namespace = binding.namespace AND revision.name = binding.name
 AND revision.revision = binding.current_revision
LEFT JOIN artifact_versions AS version ON version.version_id = revision.version_id
LEFT JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
ORDER BY binding.name`
