package artifacts

// SQL statements for postgres_query.go.

// listLineageSQL returns one page of artifact_lineage edges in which exact
// revision $1..$5 is the source or the target, newest first, optionally
// excluding lineage kind $6. $7..$10 are the keyset cursor (created_at,
// target_revision, source_revision, lineage_kind); a NULL $7 starts the first
// page. Used by PostgresRepository.ListLineage.
var listLineageSQL = `
SELECT lineage.lineage_kind,
       lineage.source_scope_kind, lineage.source_scope_id,
       lineage.source_namespace, lineage.source_name, lineage.source_revision,
       lineage.target_scope_kind, lineage.target_scope_id,
       lineage.target_namespace, lineage.target_name, lineage.target_revision,
       lineage.created_at
FROM artifact_lineage AS lineage
WHERE ((
      lineage.source_scope_kind = $1 AND lineage.source_scope_id = $2
      AND lineage.source_namespace = $3 AND lineage.source_name = $4 AND lineage.source_revision = $5
    ) OR (
      lineage.target_scope_kind = $1 AND lineage.target_scope_id = $2
      AND lineage.target_namespace = $3 AND lineage.target_name = $4 AND lineage.target_revision = $5
    ))
  AND ($6::text = '' OR lineage.lineage_kind <> $6)
  AND ($7::timestamptz IS NULL OR (
    lineage.created_at, lineage.target_revision, lineage.source_revision, lineage.lineage_kind
  ) < ($7, $8, $9, $10))
ORDER BY lineage.created_at DESC, lineage.target_revision DESC,
         lineage.source_revision DESC, lineage.lineage_kind DESC
LIMIT $11`

// listMetadataSQL returns one keyset page of the current bindings of scope
// ($1, $2) with their current revision metadata, in namespace and name order:
// only namespace $3 when it is not NULL, without namespace $4 when it is not
// NULL, without the namespaces starting with $5 when it is not empty, after
// the key ($6, $7) when $6 is not empty, without the hidden namespace $9 when
// it is not empty, at most $8 rows. Used by PostgresRepository.ListMetadata.
var listMetadataSQL = `
SELECT binding.namespace, binding.name, revision.revision,
       version.media_type, blob.size_bytes, true, binding.frozen, revision.created_at, ` + gitSourceProjection + `
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
  AND ($3::text IS NULL OR binding.namespace = $3)
  AND ($4::text IS NULL OR binding.namespace <> $4)
  AND ($5::text = '' OR left(binding.namespace, length($5)) <> $5)
  AND ($6::text = '' OR (binding.namespace, binding.name) > ($6, $7))
  AND ($9::text = '' OR binding.namespace <> $9)
ORDER BY binding.namespace, binding.name
LIMIT $8`

// listProjectMetadataSQL is listMetadataSQL for the generic listing of
// ProjectScope $1, which hides the Server-reserved audit- namespaces
// (ART-R26): only namespace $2 when it is not NULL, without namespace $3 when
// it is not NULL, after the key ($4, $5) when $4 is not empty, without the
// hidden namespace $7 when it is not empty, at most $6 rows. Its literal scope kind and namespace predicate repeat the predicate
// of artifact_bindings_project_listing_idx, so PostgreSQL scans that index of
// exactly the listable Project bindings and a page never reads an
// Audit-managed binding (ART-R115). Used by PostgresRepository.ListMetadata.
var listProjectMetadataSQL = `
SELECT binding.namespace, binding.name, revision.revision,
       version.media_type, blob.size_bytes, true, binding.frozen, revision.created_at, ` + gitSourceProjection + `
FROM artifact_bindings AS binding
JOIN artifact_binding_revisions AS revision
  ON revision.scope_kind = binding.scope_kind
 AND revision.scope_id = binding.scope_id
 AND revision.namespace = binding.namespace
 AND revision.name = binding.name
 AND revision.revision = binding.current_revision
JOIN artifact_versions AS version ON version.version_id = revision.version_id
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE binding.scope_kind = 'project' AND binding.scope_id = $1
  AND binding.namespace NOT LIKE 'audit-%'
  AND ($2::text IS NULL OR binding.namespace = $2)
  AND ($3::text IS NULL OR binding.namespace <> $3)
  AND ($4::text = '' OR (binding.namespace, binding.name) > ($4, $5))
  AND ($7::text = '' OR binding.namespace <> $7)
ORDER BY binding.namespace, binding.name
LIMIT $6`
