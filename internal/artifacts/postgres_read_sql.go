package artifacts

// SQL statements for postgres_read.go.

// readBindingSQL resolves the current revision of a binding ($1..$4: scope
// kind, scope id, namespace, name) through its version to the blob. Returns
// revision, media type, blob backend, object key, inline payload, sha256,
// size, and the binding and revision created_at timestamps.
// Used by PostgresRepository.Read.
var readBindingSQL = `
SELECT revision.revision, version.media_type, blob.backend, blob.object_key, blob.payload, blob.sha256, blob.size_bytes,
       binding.created_at, revision.created_at
FROM artifact_bindings AS binding
JOIN artifact_binding_revisions AS revision
  ON revision.scope_kind = binding.scope_kind
 AND revision.scope_id = binding.scope_id
 AND revision.namespace = binding.namespace
 AND revision.name = binding.name
 AND revision.revision = binding.current_revision
JOIN artifact_versions AS version ON version.version_id = revision.version_id
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE binding.scope_kind = $1 AND binding.scope_id = $2
  AND binding.namespace = $3 AND binding.name = $4`

// readRevisionSQL resolves exact revision $5 of a binding ($1..$4) through its
// version to the blob, returning the same columns as readBindingSQL.
// Used by PostgresRepository.Read.
var readRevisionSQL = `
SELECT revision.revision, version.media_type, blob.backend, blob.object_key, blob.payload, blob.sha256, blob.size_bytes,
       binding.created_at, revision.created_at
FROM artifact_binding_revisions AS revision
JOIN artifact_bindings AS binding
  ON binding.scope_kind = revision.scope_kind
 AND binding.scope_id = revision.scope_id
 AND binding.namespace = revision.namespace
 AND binding.name = revision.name
JOIN artifact_versions AS version ON version.version_id = revision.version_id
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE revision.scope_kind = $1 AND revision.scope_id = $2
  AND revision.namespace = $3 AND revision.name = $4 AND revision.revision = $5`
