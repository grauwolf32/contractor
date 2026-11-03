package pgstore

// SQL statements for report_review.go.

// reportReviewSelectionCTE locks the pending report-acceptance review of the
// Audit row in the named CTE while that row is in waiting_review. A report
// review can be decided only there, so every other way out of that state
// closes it: owner transitions and owner or Project deletion use this CTE,
// while a decision or ExpireReportReview records its own outcome. A report
// review starts with dispatch closed and every item settled, so the
// Controller's deadline pause and item-review activation never leave
// waiting_review while one is pending. The statement changes at most that one
// Audit, adds reportReviewExpiryEvents after its own event and follows its
// changed CTE with reportReviewClosureCTE.
func reportReviewSelectionCTE(audit string) string {
	return `report_review AS MATERIALIZED (
    SELECT review.request_id, review.audit_id, review.revision
      FROM ` + audit + ` AS leaving
      JOIN audit_review_requests AS review
        ON review.audit_id = leaving.audit_id AND review.subject_kind = 'audit-report'
     WHERE leaving.state = 'waiting_review' AND review.state = 'pending'
     FOR UPDATE OF review
)`
}

// reportReviewExpiryEvents records one review.expired event per review that
// reportReviewSelectionCTE locked, at the revision reportReviewClosureCTE
// gives it.
var reportReviewExpiryEvents = auditEventsFrom(
	"report_review AS review", "review.request_id",
	"'review.expired'", "review.request_id", "review.revision + 1",
	"jsonb_build_object('subjectKind', 'audit-report', 'kind', 'report-acceptance')",
)

// reportReviewClosureCTE expires the reviews reportReviewSelectionCTE locked
// for the Audit the changed CTE returns.
const reportReviewClosureCTE = `expired_report_review AS (
    UPDATE audit_review_requests AS review
       SET state = 'expired', revision = review.revision + 1,
           updated_at = GREATEST(clock_timestamp(), review.updated_at + interval '1 microsecond')
      FROM report_review, changed
     WHERE review.request_id = report_review.request_id
       AND review.audit_id = changed.audit_id AND review.state = 'pending'
)`

// reportAcceptedEvents record an accepted report candidate like a committed
// report.
var reportAcceptedEvents = auditEvents(auditEvent(
	"'audit.report_committed'", "audit.audit_id", "audit.revision",
	"jsonb_build_object('requestDigest', $4::text)",
))

// reportRejectedEvents record the terminal transition of a rejected report.
var reportRejectedEvents = auditEvents(auditEvent(
	"'audit.state_changed'", "audit.audit_id", "audit.revision",
	"jsonb_build_object('from', 'waiting_review', 'to', 'failed')",
))

// acceptReportCandidateSQL publishes an accepted report candidate: with Audit
// $1 in waiting_review at revision $2 and its report review matching request
// $3 and digest $4 (both locked), it inserts the two report links from $5, completes
// the Audit with dispatch closed and hold released, and appends
// audit.report_committed; the caller completes the trailing SELECT.
// Used by ReviewStore.AcceptReportCandidate.
var acceptReportCandidateSQL = `
WITH link_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($5::jsonb) AS link(
        logical_key text, artifact_ref jsonb, artifact_digest text,
        media_type text, size_bytes bigint, source_provenance jsonb, display_ref text
    )
), gate AS MATERIALIZED (
    SELECT audit.audit_id
      FROM audits AS audit
      JOIN audit_review_requests AS candidate
        ON candidate.audit_id = audit.audit_id AND candidate.subject_kind = 'audit-report'
     WHERE audit.audit_id = $1 AND audit.revision = $2
       AND audit.state = 'waiting_review'
       AND candidate.request_id = $3 AND candidate.subject_digest = $4
       AND NOT EXISTS (
           SELECT 1 FROM audit_artifact_links AS existing
            WHERE existing.audit_id = audit.audit_id
              AND existing.logical_key IN ('report/machine', 'report/summary')
       )
     FOR UPDATE OF audit, candidate
), inserted_links AS (
    INSERT INTO audit_artifact_links (
        audit_id, logical_key, artifact_ref, artifact_digest,
        media_type, size_bytes, source_provenance, display_ref
    )
    SELECT gate.audit_id, link.logical_key, link.artifact_ref,
           link.artifact_digest, link.media_type, link.size_bytes,
           link.source_provenance, link.display_ref
      FROM gate CROSS JOIN link_input AS link
    RETURNING audit_id
), changed AS (
    UPDATE audits AS audit
       SET state = 'completed', dispatch_state = 'closed', hold_state = 'released',
           stop_reason_code = CASE WHEN audit.stop_reason_code = 'round_complete'
                                   THEN NULL ELSE audit.stop_reason_code END,
           stop_reason_message = CASE WHEN audit.stop_reason_code = 'round_complete'
                                      THEN NULL ELSE audit.stop_reason_message END,
           finished_at = clock_timestamp(),
           ` + reportAcceptedEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM gate
     WHERE audit.audit_id = gate.audit_id
       AND (SELECT count(*) FROM inserted_links) = 2
    RETURNING audit.*, ` + reportAcceptedEvents.appended() + `
)
SELECT `

// rejectReportCandidateSQL fails Audit $1 when it is in waiting_review at
// revision $2 and its report review matches request $3, subject revision $4
// and digest $5 (both rows locked): dispatch closes, the hold is released, stop
// reason report_rejected is set and audit.state_changed is appended. The caller
// completes the trailing SELECT. Used by ReviewStore.RejectReportCandidate.
var rejectReportCandidateSQL = `
WITH gate AS MATERIALIZED (
    SELECT audit.audit_id
      FROM audits AS audit
      JOIN audit_review_requests AS candidate
        ON candidate.audit_id = audit.audit_id AND candidate.subject_kind = 'audit-report'
     WHERE audit.audit_id = $1 AND audit.revision = $2
       AND audit.state = 'waiting_review'
       AND candidate.request_id = $3 AND candidate.subject_revision = $4
       AND candidate.subject_digest = $5
     FOR UPDATE OF audit, candidate
), changed AS (
    UPDATE audits AS audit
       SET state = 'failed', dispatch_state = 'closed', hold_state = 'released',
           stop_reason_code = 'report_rejected',
           stop_reason_message = 'The exact proposed Audit report was rejected by its owner.',
           finished_at = clock_timestamp(),
           ` + reportRejectedEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM gate
     WHERE audit.audit_id = gate.audit_id
    RETURNING audit.*, ` + reportRejectedEvents.appended() + `
)
SELECT `
