package postgres

// SQL statements for history.go.

// lockHistoryBindingSQL locks binding ($1, $2, $3, $4) for history admission
// and returns its current revision, frozen flag, retained-history counters
// and the transaction's isolation level. Used by lockHistoryBinding.
var lockHistoryBindingSQL = `
SELECT binding.current_revision, binding.frozen, binding.retained_revisions, binding.retained_bytes,
       current_setting('transaction_isolation')
FROM artifact_bindings AS binding
JOIN artifact_scopes AS scope
  ON scope.scope_kind = binding.scope_kind AND scope.scope_id = binding.scope_id
WHERE binding.scope_kind = $1 AND binding.scope_id = $2
  AND binding.namespace = $3 AND binding.name = $4
FOR NO KEY UPDATE OF binding
FOR KEY SHARE OF scope`

// historyCandidatesSQL returns at most $7 unprotected revisions of binding
// ($1, $2, $3, $4) after the key ($5, $6), oldest first, with their versions
// and the encoded size of their blobs. It reads the eligible partial index
// and loads no payload. Used by planHistory.
var historyCandidatesSQL = `
SELECT revision.revision, revision.created_at, revision.version_id, blob.size_bytes
FROM artifact_binding_revisions AS revision
JOIN artifact_versions AS version ON version.version_id = revision.version_id
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE revision.scope_kind = $1 AND revision.scope_id = $2
  AND revision.namespace = $3 AND revision.name = $4
  AND revision.hold_count = 0
  AND (revision.created_at, revision.revision) > ($5::timestamptz, $6::text)
ORDER BY revision.created_at, revision.revision
LIMIT $7`

// lockHistoryVictimsSQL locks the planned victims $5 of binding ($1, $2, $3,
// $4) in key order, the order every holder locks revisions in, and returns
// those that are still unprotected. Used by lockHistoryVictims.
var lockHistoryVictimsSQL = `
SELECT revision.revision
FROM artifact_binding_revisions AS revision
WHERE revision.scope_kind = $1 AND revision.scope_id = $2
  AND revision.namespace = $3 AND revision.name = $4
  AND revision.revision = ANY($5::text[])
  AND revision.hold_count = 0
ORDER BY revision.revision
FOR UPDATE OF revision`

// deleteHistoryVictimsSQL removes the locked victims $5 of binding ($1, $2,
// $3, $4) under the retention exception and returns their versions. Used by
// removeHistoryVictims.
var deleteHistoryVictimsSQL = `
DELETE FROM artifact_binding_revisions AS revision
WHERE revision.scope_kind = $1 AND revision.scope_id = $2
  AND revision.namespace = $3 AND revision.name = $4
  AND revision.revision = ANY($5::text[])
  AND revision.hold_count = 0
RETURNING revision.version_id`

// releaseHistoryCountersSQL subtracts $5 revisions and $6 bytes that a
// maintenance pass removed from binding ($1, $2, $3, $4). Used by
// Repository.maintainHistory.
var releaseHistoryCountersSQL = `
UPDATE artifact_bindings
SET retained_revisions = retained_revisions - $5, retained_bytes = retained_bytes - $6
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4`
