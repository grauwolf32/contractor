package credentials

import (
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// SQL statements for runtime_repository.go. JSON members and paths come from
// the constants beside the encoded types, so the text matches the indexes.

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
WHERE c.canonical_document::jsonb #>> '` + runtimeconfig.WorkerTelemetryCredentialPath + `' = $1
   OR c.canonical_document::jsonb #>> '` + runtimeconfig.WorkerHTTPProxyCredentialPath + `' = $1
   OR c.canonical_document::jsonb #>> '` + runtimeconfig.WorkerCaidoCredentialPath + `' = $1
   OR c.canonical_document::jsonb #>> '` + runtimeconfig.PlannerTelemetryCredentialPath + `' = $1
ORDER BY b.label
LIMIT $2`

// projectTargetCredentialPath is the credential ID of a stored Project HTTP
// target snapshot, the expression of
// workflow_runs_project_http_target_credential_idx.
const projectTargetCredentialPath = `{` + runtimesettings.CredentialMember + `,` +
	runtimesettings.CredentialIDMember + `}`

// runtimeCredentialRunUsageSQL lists active Runs (up to $2, oldest first)
// whose immutable snapshots pin credential $1: the RuntimeConfig snapshot's
// runtimeCredentialIds or the Project HTTP target credential. The redundant
// "? 'credential'" test lets the planner prove the predicate of
// workflow_runs_project_http_target_credential_idx, so both arms are index
// scans over active Runs instead of a scan of the Run history.
// Used by RuntimeCredentialRepository.InspectRuntimeCredentialUsage.
var runtimeCredentialRunUsageSQL = `
SELECT run_id
FROM workflow_runs
WHERE state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling')
  AND (
      (runtime_config_snapshot->'` + runtimeconfig.RuntimeCredentialIDsMember + `') ? $1
      OR (
          project_http_target_snapshot ? '` + runtimesettings.CredentialMember + `'
          AND project_http_target_snapshot#>>'` + projectTargetCredentialPath + `' = $1
      )
  )
ORDER BY created_at, run_id
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
      a.runtime_configuration->'` + runstore.AllocationProvenanceMember + `'->'` +
	runtimesettings.RuntimeCredentialRefsMember + `'
          @> jsonb_build_array(jsonb_build_object('` + runtimesettings.CredentialIDMember + `', $1::text))
      OR r.project_http_target_snapshot#>>'` + projectTargetCredentialPath + `' = $1
  )
ORDER BY a.created_at, a.allocation_id
LIMIT $2`
