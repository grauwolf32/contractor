package scheduler

// SQL statements for postgres.go.

// verifyRequiredWorkflowOutputsSQL returns the media type of the Artifact
// version at the current revision of Run $1's binding $2 in the 'outputs'
// namespace. No row means the output is not bound.
// Used by verifyRequiredWorkflowOutputs.
var verifyRequiredWorkflowOutputsSQL = `
SELECT version.media_type
FROM artifact_bindings AS binding
JOIN artifact_binding_revisions AS revision
  ON revision.scope_kind = binding.scope_kind
 AND revision.scope_id = binding.scope_id
 AND revision.namespace = binding.namespace
 AND revision.name = binding.name
 AND revision.revision = binding.current_revision
JOIN artifact_versions AS version ON version.version_id = revision.version_id
WHERE binding.scope_kind = 'run' AND binding.scope_id = $1
  AND binding.namespace = 'outputs' AND binding.name = $2`
