package artifacts

// SQL statements for postgres_read_batch.go.

// readExactBatchSQL resolves exact revisions, passed as parallel arrays
// $1..$5, to their versions and blobs in request order, tagging each row with
// its 1-based request ordinal; a missing reference yields no row. Inline
// payloads are returned only when the combined blob size is within $6, and that
// total is returned on every row. Used by PostgresRepository.ReadExactBatch.
var readExactBatchSQL = `
SELECT input.ordinality, revision.revision, version.media_type,
       blob.backend, blob.object_key,
       CASE WHEN sum(blob.size_bytes) OVER () <= $6 THEN blob.payload END,
       blob.sha256, blob.size_bytes, binding.created_at, revision.created_at,
       sum(blob.size_bytes) OVER ()
  FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::text[])
       WITH ORDINALITY AS input(scope_kind, scope_id, namespace, name, revision, ordinality)
  JOIN artifact_binding_revisions AS revision
    ON revision.scope_kind = input.scope_kind AND revision.scope_id = input.scope_id
   AND revision.namespace = input.namespace AND revision.name = input.name
   AND revision.revision = input.revision
  JOIN artifact_bindings AS binding
    ON binding.scope_kind = revision.scope_kind AND binding.scope_id = revision.scope_id
   AND binding.namespace = revision.namespace AND binding.name = revision.name
  JOIN artifact_versions AS version ON version.version_id = revision.version_id
  JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
 ORDER BY input.ordinality`
