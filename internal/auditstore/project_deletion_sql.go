package auditstore

// SQL statements for project_deletion.go.

// requestProjectOwnedDeletionSQL is the head of a statement completed by
// deletionRequestCTEs. It locks owner $2's Project $1 while it is deleting
// under deletion claim $3 in phase $4, then locks (SKIP LOCKED) that Project's
// oldest Audit without a deletion request as the candidate.
// Used by PostgresStore.RequestProjectOwnedDeletion.
var requestProjectOwnedDeletionSQL = `
WITH live_project AS MATERIALIZED (
    SELECT project_id
      FROM projects
     WHERE project_id = $1 AND owner_id = $2
       AND lifecycle_state = 'deleting' AND deletion_phase = $4
       AND deletion_claim_id = $3
     FOR UPDATE
), candidate AS MATERIALIZED (
    SELECT audit.*
      FROM audits AS audit JOIN live_project USING (project_id)
     WHERE audit.deletion_requested_at IS NULL
     ORDER BY audit.created_at, audit.audit_id
     FOR UPDATE OF audit SKIP LOCKED
     LIMIT 1
), `
