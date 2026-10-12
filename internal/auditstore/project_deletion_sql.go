package auditstore

// SQL statements for project_deletion.go.

// requestProjectOwnedDeletionSQL is the head of a statement completed by
// deletionRequestCTEs. It locks (SKIP LOCKED) the oldest Audit of Project $1
// without a deletion request as the candidate. The caller holds the lock of
// the deleting Project under its deletion claim.
// Used by PostgresStore.RequestProjectOwnedDeletion.
var requestProjectOwnedDeletionSQL = `
WITH candidate AS MATERIALIZED (
    SELECT audit.*
      FROM audits AS audit
     WHERE audit.project_id = $1 AND audit.deletion_requested_at IS NULL
     ORDER BY audit.created_at, audit.audit_id
     FOR UPDATE OF audit SKIP LOCKED
     LIMIT 1
), `
