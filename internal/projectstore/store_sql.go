package projectstore

// SQL statements for store.go.

// listedProjectSQL is the condition that the Project row aliased project is
// listed to its owner: it is not the execution workspace of one of the
// owner's eval submissions. It is the one definition of that rule for the
// Project list and, through OwnerListedProjectSQL, the owner-wide lists.
func listedProjectSQL(project string) string {
	return `NOT EXISTS (
      SELECT 1 FROM eval_submissions AS submission
      JOIN eval_experiments AS experiment USING (experiment_id)
      WHERE experiment.owner_id = ` + project + `.owner_id
        AND submission.execution_project_id = ` + project + `.project_id)`
}

// OwnerListedProjectSQL is the condition that the row whose Project and owner
// are the columns projectID and ownerID belongs to a Project listed to its
// owner that is a regular Project (not an evaluation Project), not deleting
// and not an eval submission's execution workspace. The owner-wide Audit,
// finding and review lists show only such rows.
func OwnerListedProjectSQL(projectID, ownerID string) string {
	return `EXISTS (
      SELECT 1 FROM projects AS listed_project
       WHERE listed_project.project_id = ` + projectID + `
         AND listed_project.owner_id = ` + ownerID + `
         AND listed_project.kind = 'project' AND listed_project.lifecycle_state = 'active'
         AND ` + listedProjectSQL("listed_project") + `)`
}

// listProjectsSQL returns one page of an owner's projects, newest first,
// optionally filtered by kind ($2). Workspace Projects owned by an eval
// submission are hidden. $3/$4 are the (created_at, project_id)
// keyset cursor of the previous page; a NULL $3 starts at the first page.
// Used by PostgresStore.List.
var listProjectsSQL = `
SELECT project_id, owner_id, kind, name, description,
       http_target_url, http_target_credential_id, http_target_credential_kind, active_checks,
       lifecycle_state, deletion_phase, deletion_requested_at,
       revision, created_at, updated_at
FROM projects
WHERE owner_id = $1
  AND ($2::text IS NULL OR kind = $2)
  AND ` + listedProjectSQL("projects") + `
  AND ($3::timestamptz IS NULL OR (created_at, project_id) < ($3, $4))
ORDER BY created_at DESC, project_id DESC
LIMIT $5`

// updateProjectSQL replaces a project's name, description and HTTP target
// under CAS on revision ($8), only while lifecycle_state is 'active'. It bumps
// the revision and a strictly increasing updated_at and returns the new row; no
// row means missing, deleting or stale, which the caller then resolves.
// Used by PostgresStore.Update.
const projectUpdateChangedSQL = `ROW(name, description, http_target_url, http_target_credential_id, http_target_credential_kind, active_checks)
 IS DISTINCT FROM ROW($1::text, $2::text, $3::text, $4::text, $5::text, COALESCE($9::text, active_checks))`

var updateProjectSQL = `
UPDATE projects
SET name = $1, description = $2, active_checks = COALESCE($9, active_checks),
    http_target_url = $3, http_target_credential_id = $4, http_target_credential_kind = $5,
    revision = revision + CASE WHEN ` + projectUpdateChangedSQL + ` THEN 1 ELSE 0 END,
    updated_at = CASE WHEN ` + projectUpdateChangedSQL + `
      THEN GREATEST(clock_timestamp(), updated_at + interval '1 microsecond') ELSE updated_at END
WHERE owner_id = $6 AND project_id = $7 AND revision = $8
  AND lifecycle_state = 'active'
RETURNING project_id, owner_id, kind, name, description,
          http_target_url, http_target_credential_id, http_target_credential_kind, active_checks,
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
          http_target_url, http_target_credential_id, http_target_credential_kind, active_checks,
          lifecycle_state, deletion_phase, deletion_requested_at,
          revision, created_at, updated_at`
