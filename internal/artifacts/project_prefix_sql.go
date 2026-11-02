package artifacts

// listProjectPrefixMetadataSQL seeks the prefix and cursor in one C-collated
// index range. Only a bounded selected page joins current revision metadata.
var listProjectPrefixMetadataSQL = `
WITH selected AS MATERIALIZED (
 SELECT * FROM artifact_bindings
 WHERE scope_kind='project' AND scope_id=$1 AND namespace=$2
 AND name COLLATE "C">=$3::text COLLATE "C"
 AND name COLLATE "C"<($3::text||'~') COLLATE "C"
 AND name COLLATE "C">$4::text COLLATE "C"
 ORDER BY name COLLATE "C" LIMIT $5
)
SELECT binding.namespace,binding.name,revision.revision,version.media_type,blob.size_bytes,
 true,binding.frozen,binding.created_at,revision.created_at,scope.created_at,blob.created_at,version.created_at,binding.updated_at,` + gitSourceProjection + `
FROM selected AS binding
JOIN artifact_scopes AS scope ON scope.scope_kind=binding.scope_kind AND scope.scope_id=binding.scope_id
JOIN artifact_binding_revisions AS revision ON revision.scope_kind=binding.scope_kind AND revision.scope_id=binding.scope_id
 AND revision.namespace=binding.namespace AND revision.name=binding.name AND revision.revision=binding.current_revision
JOIN artifact_versions AS version ON version.version_id=revision.version_id
JOIN artifact_blobs AS blob ON blob.sha256=version.blob_sha256
ORDER BY binding.name COLLATE "C"`
