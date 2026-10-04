package auditstore

// SQL statements for report_review.go.

// proposeReportSQL freezes report links $9/$10 as the candidate of a 30-day
// review request $7 under the live controller claim ($1-$3). The Audit at
// revision $4 must be finalizing with round $5 closed at revision $6, no
// outstanding Runs, all items settled, all executions collected and no report
// links; it moves to waiting_review and appends review.requested. The caller
// completes the trailing SELECT. Used by PostgresStore.ProposeReport.
var proposeReportSQL = `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), gate AS MATERIALIZED (
    SELECT audit.*, round.revision AS round_revision
      FROM audits AS audit
      JOIN live_claim USING (audit_id)
      JOIN audit_rounds AS round
        ON round.audit_id = audit.audit_id AND round.round_id = $5
     WHERE audit.audit_id = $1 AND audit.revision = $4
       AND audit.state = 'finalizing' AND audit.dispatch_state = 'closed'
       AND audit.current_round_id = $5 AND audit.outstanding_run_count = 0
       AND round.revision = $6 AND round.state = 'closed'
       AND NOT EXISTS (
           SELECT 1 FROM audit_items AS item
            WHERE item.audit_id = audit.audit_id AND item.state <> 'settled'
       )
       AND NOT EXISTS (
           SELECT 1 FROM audit_executions AS execution
            WHERE execution.audit_id = audit.audit_id AND execution.state <> 'collected'
       )
       AND NOT EXISTS (
           SELECT 1 FROM audit_artifact_links AS existing
            WHERE existing.audit_id = audit.audit_id
              AND existing.logical_key IN ('report/machine', 'report/summary')
       )
     FOR UPDATE OF audit, round
), inserted_request AS (
    INSERT INTO audit_review_requests (
        request_id, audit_id, finding_id, subject_kind, subject_id, kind,
        subject_revision, subject_digest, requested_actions, state,
        expires_at, idempotency_key, request_digest
    )
    SELECT $7, audit_id, NULL, 'audit-report', audit_id,
           'report-acceptance', $4, $8, '["approve","reject"]'::jsonb,
           'pending', clock_timestamp() + interval '30 days',
           'auto-report:' || substr($8, 8, 64), $8
      FROM gate
    RETURNING request_id, audit_id
), inserted_candidate AS (
    INSERT INTO audit_report_candidates (
        audit_id, request_id, round_id, subject_revision, subject_digest,
        machine_link, summary_link
    )
    SELECT request.audit_id, request.request_id, $5, $4, $8,
           $9::jsonb, $10::jsonb
      FROM inserted_request AS request
    RETURNING audit_id
), changed AS (
    UPDATE audits AS audit
       SET state = 'waiting_review', revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM inserted_candidate AS candidate
     WHERE audit.audit_id = candidate.audit_id
    RETURNING audit.*
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT audit_id, next_event_sequence - 1, 'review.requested', $7,
           1, jsonb_build_object(
               'subjectKind', 'audit-report', 'subjectId', audit_id,
               'kind', 'report-acceptance'
           )
      FROM changed
)
SELECT `

// expireReportReviewSQL fails a waiting_review Audit at revision $4 whose
// report-acceptance review is expired or pending past expires_at (database
// time), under the live controller claim ($1-$3). It expires the request,
// closes dispatch, releases the hold, sets stop reason
// report_acceptance_expired, appends review.expired and returns whether the
// Audit changed. Used by PostgresStore.ExpireReportReview.
var expireReportReviewSQL = `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), gate AS MATERIALIZED (
    SELECT audit.audit_id, audit.next_event_sequence, review.request_id,
           review.state AS review_state
      FROM audits AS audit
      JOIN live_claim USING (audit_id)
      JOIN audit_report_candidates AS candidate USING (audit_id)
      JOIN audit_review_requests AS review
        ON review.request_id = candidate.request_id
       AND review.audit_id = candidate.audit_id
     WHERE audit.audit_id = $1 AND audit.revision = $4
       AND audit.state = 'waiting_review'
       AND (
           review.state = 'expired'
           OR (review.state = 'pending' AND review.expires_at <= clock_timestamp())
       )
     FOR UPDATE OF audit, review
), expired_request AS (
    UPDATE audit_review_requests AS review
       SET state = 'expired', revision = review.revision + 1,
           updated_at = GREATEST(clock_timestamp(), review.updated_at + interval '1 microsecond')
      FROM gate
     WHERE review.request_id = gate.request_id AND review.state = 'pending'
    RETURNING review.request_id
), terminal AS (
    UPDATE audits AS audit
       SET state = 'failed', dispatch_state = 'closed', hold_state = 'released',
           stop_reason_code = 'report_acceptance_expired',
           stop_reason_message = 'The exact Audit report was not accepted before its review expired.',
           revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond'),
           finished_at = clock_timestamp()
      FROM gate
     WHERE audit.audit_id = gate.audit_id
    RETURNING audit.audit_id, audit.next_event_sequence, gate.request_id
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, summary
    )
    SELECT audit_id, next_event_sequence - 1, 'review.expired', request_id,
           jsonb_build_object(
               'subjectKind', 'audit-report',
               'kind', 'report-acceptance',
               'terminalState', 'failed'
           )
      FROM terminal
)
SELECT EXISTS (SELECT 1 FROM terminal)`

// acceptReportCandidateSQL publishes an accepted report candidate: with Audit
// $1 in waiting_review at revision $2 and its candidate matching request $3 and
// digest $4 (both locked), it inserts the two report links from $5, completes
// the Audit with dispatch closed and hold released, and appends
// audit.report_committed; the caller completes the trailing SELECT.
// Used by PostgresStore.AcceptReportCandidate.
var acceptReportCandidateSQL = `
WITH link_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($5::jsonb) AS link(
        logical_key text, artifact_ref jsonb, artifact_digest text,
        media_type text, size_bytes bigint, source_provenance jsonb, display_ref text
    )
), gate AS MATERIALIZED (
    SELECT audit.audit_id
      FROM audits AS audit
      JOIN audit_report_candidates AS candidate USING (audit_id)
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
           revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM gate
     WHERE audit.audit_id = gate.audit_id
       AND (SELECT count(*) FROM inserted_links) = 2
    RETURNING audit.*
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, entity_revision, summary)
    SELECT audit_id, next_event_sequence - 1, 'audit.report_committed', audit_id, revision,
           jsonb_build_object('requestDigest', $4::text)
      FROM changed
)
SELECT `

// rejectReportCandidateSQL fails Audit $1 when it is in waiting_review at
// revision $2 and its report candidate matches request $3, subject revision $4
// and digest $5 (both rows locked): dispatch closes, the hold is released, stop
// reason report_rejected is set and audit.state_changed is appended. The caller
// completes the trailing SELECT. Used by PostgresStore.RejectReportCandidate.
var rejectReportCandidateSQL = `
WITH gate AS MATERIALIZED (
    SELECT audit.audit_id
      FROM audits AS audit
      JOIN audit_report_candidates AS candidate USING (audit_id)
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
           revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM gate
     WHERE audit.audit_id = gate.audit_id
    RETURNING audit.*
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, entity_revision, summary)
    SELECT audit_id, next_event_sequence - 1, 'audit.state_changed', audit_id, revision,
           jsonb_build_object('from', 'waiting_review', 'to', 'failed')
      FROM changed
)
SELECT `
