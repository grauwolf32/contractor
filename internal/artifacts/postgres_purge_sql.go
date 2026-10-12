package artifacts

// SQL statements for postgres_purge.go. A purge walks the binding revisions of
// its purge set in primary-key order, at most purgeBatchSize per batch. Each
// batch is the key range (after, last]: the scope row lock keeps every other
// writer out of the scope, so the range holds exactly the selected revisions,
// and every range delete starts its index scan behind the rows earlier
// batches of the same transaction already deleted.

// purgeScopeRevisionBatchSQL returns the next batch of binding revisions of
// scope ($1, $2) after the key ($3, $4, $5), in key order, at most $6 rows,
// with their versions. Used by PostgresPurger.purgeRevisionBatch.
var purgeScopeRevisionBatchSQL = `
SELECT namespace, name, revision, version_id
FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2
  AND (namespace, name, revision) > ($3, $4, $5)
ORDER BY namespace, name, revision
LIMIT $6`

// purgeNamespaceRevisionBatchSQL is purgeScopeRevisionBatchSQL restricted to
// namespace $3, after the key ($4, $5) within it. Used by
// PostgresPurger.purgeRevisionBatch for an Audit namespace.
var purgeNamespaceRevisionBatchSQL = `
SELECT namespace, name, revision, version_id
FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3
  AND (name, revision) > ($4, $5)
ORDER BY name, revision
LIMIT $6`

// purgeTargetLineageSQL deletes the lineage edges of scope ($1, $2) whose
// target revision lies in the key range (($3, $4, $5), ($6, $7, $8)]. Used by
// PostgresPurger.purgeRevisionBatch.
var purgeTargetLineageSQL = `
DELETE FROM artifact_lineage
WHERE target_scope_kind = $1 AND target_scope_id = $2
  AND (target_namespace, target_name, target_revision) > ($3, $4, $5)
  AND (target_namespace, target_name, target_revision) <= ($6, $7, $8)`

// purgeSourceLineageSQL deletes the lineage edges, in any target scope, whose
// source revision of scope ($1, $2) lies in the key range
// (($3, $4, $5), ($6, $7, $8)]. Used by PostgresPurger.purgeRevisionBatch.
var purgeSourceLineageSQL = `
DELETE FROM artifact_lineage
WHERE source_scope_kind = $1 AND source_scope_id = $2
  AND (source_namespace, source_name, source_revision) > ($3, $4, $5)
  AND (source_namespace, source_name, source_revision) <= ($6, $7, $8)`

// purgeBindingSQL deletes the bindings of scope ($1, $2) whose current
// revision lies in the key range (($3, $4, $5), ($6, $7, $8)], so no binding
// outlives its current revision even when the batch commits on its own. Used
// by PostgresPurger.purgeRevisionBatch.
var purgeBindingSQL = `
DELETE FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2
  AND (namespace, name) >= ($3, $4) AND (namespace, name) <= ($6, $7)
  AND (namespace, name, current_revision) > ($3, $4, $5)
  AND (namespace, name, current_revision) <= ($6, $7, $8)`

// purgeRevisionSQL deletes the binding revisions of scope ($1, $2) in the key
// range (($3, $4, $5), ($6, $7, $8)] after their lineage edges and bindings
// are gone. Used by PostgresPurger.purgeRevisionBatch.
var purgeRevisionSQL = `
DELETE FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2
  AND (namespace, name, revision) > ($3, $4, $5)
  AND (namespace, name, revision) <= ($6, $7, $8)`

// purgeLockVersionsSQL locks the candidate versions $1 in byte order, the
// order collectUnreferencedVersions chunks them in. Used by
// PostgresPurger.collectUnreferencedVersions.
var purgeLockVersionsSQL = `
SELECT version_id FROM artifact_versions
WHERE version_id = ANY($1::text[])
ORDER BY version_id COLLATE "C"
FOR UPDATE`

// purgeUnreferencedVersionsSQL deletes the candidate versions $1 that no
// binding revision references any more and returns their blob digests. Used by
// PostgresPurger.collectUnreferencedVersions.
var purgeUnreferencedVersionsSQL = `
DELETE FROM artifact_versions AS version
WHERE version.version_id = ANY($1::text[])
  AND NOT EXISTS (
      SELECT 1
      FROM artifact_binding_revisions AS revision
      WHERE revision.version_id = version.version_id
  )
RETURNING version.blob_sha256`

// purgeUnreferencedBlobsSQL deletes the candidate blobs $1 that no version
// references any more and returns their backend and filesystem object key.
// Used by PostgresPurger.collectUnreferencedBlobs.
var purgeUnreferencedBlobsSQL = `
DELETE FROM artifact_blobs AS blob
WHERE blob.sha256 = ANY($1::bytea[])
  AND NOT EXISTS (
      SELECT 1 FROM artifact_versions AS version WHERE version.blob_sha256 = blob.sha256
  )
RETURNING backend, object_key`
