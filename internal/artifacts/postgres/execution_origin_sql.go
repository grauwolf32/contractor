package postgres

// executionOriginSQL resolves exactly one authorized binding revision. Only
// Server-assigned version metadata supplies provenance, never Artifact bytes.
const executionOriginSQL = `
SELECT version.version_id, version.execution_origin
FROM artifact_binding_revisions AS revision
JOIN artifact_versions AS version ON version.version_id=revision.version_id
WHERE revision.scope_kind=$1 AND revision.scope_id=$2
 AND revision.namespace=$3 AND revision.name=$4 AND revision.revision=$5`
