package pgstore

import (
	"github.com/grauwolf32/contractor/internal/auditstore"
)

// SQL statements for deletion.go and project_deletion.go.

// requestDeleteSQL records owner $1's deletion request for Audit $2 at
// revision $3 through deletionRequestCTEs and stores the audit.delete
// idempotency record ($4-$6). An Audit that is already deleting is returned
// unchanged with the new record. Returns the Audit columns.
// Used by OwnerStore.RequestDelete.
var requestDeleteSQL = `
WITH candidate AS MATERIALIZED (
    SELECT audit.*
      FROM audits AS audit
     WHERE audit.owner_id = $1 AND audit.audit_id = $2 AND audit.revision = $3
     FOR UPDATE
), ` + deletionRequestCTEs(auditDeletionByOwner) + `, selected AS MATERIALIZED (
    SELECT ` + prefixedAuditColumns("changed") + ` FROM changed
    UNION ALL
    SELECT ` + prefixedAuditColumns("candidate") + `
      FROM candidate WHERE candidate.state = 'deleting'
), idempotency_row AS (
    INSERT INTO audit_idempotency (
        owner_id, operation, idempotency_key, request_digest,
        audit_id, resource_id, response_snapshot
    )
    SELECT $1, ` + operationLiteral(auditstore.MutationDelete) + `, $4, $5, audit_id, audit_id, $6::jsonb
      FROM selected
)
SELECT * FROM selected`

// deletionRequestEvents record a deletion request, naming a Project deletion
// as its source, and expire a pending report review after it.
func deletionRequestEvents(source auditDeletionSource) auditEventBatch {
	summary := "jsonb_build_object('state', audit.state)"
	if source == auditDeletionByProject {
		summary = "jsonb_build_object('state', audit.state, 'source', 'project-deletion')"
	}
	return auditEvents(
		auditEvent("'audit.delete_requested'", "audit.audit_id", "audit.revision", summary),
		reportReviewExpiryEvents,
	)
}

// deletionRequestCTEs records a deletion request for the Audit that the
// statement's preceding candidate CTE locks with every audits column. Active
// work first enters cancelling; drafts and terminal Audits enter deleting
// directly, and leaving waiting_review closes the pending report review. An
// Audit that is already deleting is not changed, leaving changed empty.
func deletionRequestCTEs(source auditDeletionSource) string {
	stopReason := `stop_reason_code = 'delete_requested',
           stop_reason_message = 'Audit deletion was requested by its owner.'`
	if source == auditDeletionByProject {
		stopReason = `stop_reason_code = 'project_deleting',
           stop_reason_message = 'The owning Project is being deleted.'`
	}
	events := deletionRequestEvents(source)
	return reportReviewSelectionCTE("candidate") + `, changed AS (
    UPDATE audits AS audit
       SET state = CASE
               WHEN candidate.state IN ('draft', 'completed', 'cancelled', 'failed') THEN 'deleting'
               ELSE 'cancelling'
           END,
           dispatch_state = 'closed',
           deletion_requested_at = COALESCE(candidate.deletion_requested_at, clock_timestamp()),
           ` + stopReason + `,
           ` + events.advance() + `,
           updated_at = GREATEST(clock_timestamp(), candidate.updated_at + interval '1 microsecond')
      FROM candidate
     WHERE audit.audit_id = candidate.audit_id AND candidate.state <> 'deleting'
    RETURNING audit.*, ` + events.appended() + `
), ` + reportReviewClosureCTE
}
