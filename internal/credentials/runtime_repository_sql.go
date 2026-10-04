package credentials

// SQL statements for runtime_repository.go.

// runtimeCredentialLabelUsageSQL lists runtime labels (up to $2, by label)
// bound to a runtime config version, matched on name, version and digest,
// whose canonical document names credential $1 as the worker telemetry, HTTP
// proxy or Caido credential, or as the planner telemetry credential.
// Used by RuntimeCredentialRepository.InspectRuntimeCredentialUsage.
var runtimeCredentialLabelUsageSQL = `
SELECT b.label
FROM runtime_label_bindings b
JOIN runtime_config_versions c
  ON c.name = b.config_name
 AND c.version = b.config_version
 AND c.digest = b.config_digest
WHERE c.canonical_document::jsonb #>> '{spec,worker,telemetry,credential}' = $1
   OR c.canonical_document::jsonb #>> '{spec,worker,httpProxy,credential}' = $1
   OR c.canonical_document::jsonb #>> '{spec,worker,caido,credential}' = $1
   OR c.canonical_document::jsonb #>> '{spec,planner,telemetry,credential}' = $1
ORDER BY b.label
LIMIT $2`

// runtimeCredentialAllocationUsageSQL lists unreleased stage allocations (up
// to $2, oldest first) that reference credential $1, either in the provenance
// runtimeCredentialRefs of their runtime configuration or through the HTTP
// target credential snapshotted on the owning workflow run.
// Used by RuntimeCredentialRepository.InspectRuntimeCredentialUsage.
var runtimeCredentialAllocationUsageSQL = `
SELECT a.allocation_id
FROM stage_allocations a
JOIN stage_executions e ON e.stage_execution_id = a.stage_execution_id
JOIN workflow_runs r ON r.run_id = e.run_id
WHERE a.release_completed_at IS NULL
  AND (
      a.runtime_configuration->'provenance'->'runtimeCredentialRefs'
          @> jsonb_build_array(jsonb_build_object('credentialId', $1::text))
      OR r.project_http_target_snapshot#>>'{credential,credentialId}' = $1
  )
ORDER BY a.created_at, a.allocation_id
LIMIT $2`
