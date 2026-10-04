package projectstore

// SQL statements for store.go.

// listProjectsSQL returns one page of an owner's projects, newest first,
// optionally filtered by kind ($2). Projects listed in
// eval_project_dependencies are hidden. $3/$4 are the (created_at, project_id)
// keyset cursor of the previous page; a NULL $3 starts at the first page.
// Used by PostgresStore.List.
var listProjectsSQL = `
SELECT project_id, owner_id, kind, name, description,
       http_target_url, http_target_credential_id, http_target_credential_kind,
       lifecycle_state, deletion_phase, deletion_requested_at,
       revision, created_at, updated_at
FROM projects
WHERE owner_id = $1
  AND ($2::text IS NULL OR kind = $2)
  AND NOT EXISTS (
      SELECT 1 FROM eval_project_dependencies d
      WHERE d.owner_id = projects.owner_id AND d.project_id = projects.project_id
  )
  AND ($3::timestamptz IS NULL OR (created_at, project_id) < ($3, $4))
ORDER BY created_at DESC, project_id DESC
LIMIT $5`

// updateProjectSQL replaces a project's name, description and HTTP target
// under CAS on revision ($8), only while lifecycle_state is 'active'. It bumps
// the revision and a strictly increasing updated_at and returns the new row; no
// row means missing, deleting or stale, which the caller then resolves.
// Used by PostgresStore.Update.
var updateProjectSQL = `
UPDATE projects
SET name = $1, description = $2,
    http_target_url = $3, http_target_credential_id = $4, http_target_credential_kind = $5,
    revision = revision + 1,
    updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
WHERE owner_id = $6 AND project_id = $7 AND revision = $8
  AND lifecycle_state = 'active'
RETURNING project_id, owner_id, kind, name, description,
          http_target_url, http_target_credential_id, http_target_credential_kind,
          lifecycle_state, deletion_phase, deletion_requested_at,
          revision, created_at, updated_at`

// beginProjectDeletionSQL moves an active project at expected revision $3 to
// lifecycle_state 'deleting' with deletion_phase 'cancelling', stamps
// deletion_requested_at and bumps the revision, fencing further updates. It
// returns the new row; on no row the caller re-reads the project to tell an
// idempotent retry from a stale revision. Used by PostgresStore.BeginDeletion.
var beginProjectDeletionSQL = `
UPDATE projects
SET lifecycle_state = 'deleting',
    deletion_phase = 'cancelling',
    deletion_requested_at = clock_timestamp(),
    revision = revision + 1,
    updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
WHERE owner_id = $1 AND project_id = $2 AND revision = $3
  AND lifecycle_state = 'active'
RETURNING project_id, owner_id, kind, name, description,
          http_target_url, http_target_credential_id, http_target_credential_kind,
          lifecycle_state, deletion_phase, deletion_requested_at,
          revision, created_at, updated_at`
