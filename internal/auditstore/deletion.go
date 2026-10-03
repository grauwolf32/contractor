package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// RequestDelete durably records deletion independently from the stop reason.
// Active work first enters cancelling; drafts and terminal Audits can enter
// deleting directly. A new key against an already-deleting exact revision is
// accepted without manufacturing another lifecycle revision.
func (s *PostgresStore) RequestDelete(
	ctx context.Context, params DeleteParams,
) (Audit, bool, error) {
	if err := validateDelete(params); err != nil {
		return Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, string(MutationDelete), params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	response, _ := json.Marshal(map[string]string{"auditId": params.AuditID})
	audit, err := scanAudit(s.db.QueryRow(ctx, `
WITH candidate AS MATERIALIZED (
    SELECT audit.*
      FROM audits AS audit
     WHERE audit.owner_id = $1 AND audit.audit_id = $2 AND audit.revision = $3
     FOR UPDATE
), report_review AS MATERIALIZED (
    SELECT review.request_id, review.audit_id
      FROM audit_report_candidates AS report
      JOIN audit_review_requests AS review
        ON review.request_id = report.request_id
       AND review.audit_id = report.audit_id
      JOIN candidate ON candidate.audit_id = report.audit_id
     WHERE candidate.state <> 'deleting' AND review.state = 'pending'
     FOR UPDATE OF review
), report_review_count AS MATERIALIZED (
    SELECT count(*)::bigint AS expired FROM report_review
), changed AS (
    UPDATE audits AS audit
       SET state = CASE
               WHEN candidate.state IN ('draft', 'completed', 'cancelled', 'failed') THEN 'deleting'
               ELSE 'cancelling'
           END,
           dispatch_state = 'closed',
           deletion_requested_at = COALESCE(candidate.deletion_requested_at, clock_timestamp()),
           stop_reason_code = 'delete_requested',
           stop_reason_message = 'Audit deletion was requested by its owner.',
           revision = candidate.revision + 1 + report_review_count.expired,
           next_event_sequence = candidate.next_event_sequence + 1 + report_review_count.expired,
           updated_at = GREATEST(clock_timestamp(), candidate.updated_at + interval '1 microsecond')
      FROM candidate, report_review_count
     WHERE audit.audit_id = candidate.audit_id AND candidate.state <> 'deleting'
    RETURNING audit.*
), expired_report_review AS (
    UPDATE audit_review_requests AS review
       SET state = 'expired', revision = review.revision + 1,
           updated_at = GREATEST(clock_timestamp(), review.updated_at + interval '1 microsecond')
      FROM report_review, changed
     WHERE review.request_id = report_review.request_id
       AND review.audit_id = changed.audit_id AND review.state = 'pending'
    RETURNING review.request_id, review.audit_id, review.revision
), selected AS MATERIALIZED (
    SELECT * FROM changed
    UNION ALL
    SELECT * FROM candidate WHERE candidate.state = 'deleting'
), idempotency_row AS (
    INSERT INTO audit_idempotency (
        owner_id, operation, idempotency_key, request_digest,
        audit_id, resource_id, response_snapshot
    )
    SELECT $1, 'audit.delete', $4, $5, audit_id, audit_id, $6::jsonb
      FROM selected
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT audit_id, next_event_sequence - 1 - report_review_count.expired,
           'audit.delete_requested', audit_id, revision,
           jsonb_build_object('state', state)
      FROM changed CROSS JOIN report_review_count
), expired_report_event AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, entity_revision, summary)
    SELECT review.audit_id,
           changed.next_event_sequence - 1 - report_review_count.expired
               + row_number() OVER (ORDER BY review.request_id),
           'review.expired', review.request_id, review.revision,
           jsonb_build_object('subjectKind', 'audit-report', 'kind', 'report-acceptance')
      FROM expired_report_review AS review
      JOIN changed USING (audit_id)
      CROSS JOIN report_review_count
)
SELECT `+prefixedAuditColumns("selected")+` FROM selected`,
		params.OwnerID, params.AuditID, params.ExpectedRevision,
		params.IdempotencyKey, params.RequestDigest, response,
	))
	if err == nil {
		return audit, true, nil
	}
	if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateUniqueViolation {
		return s.replayAudit(ctx, params.OwnerID, string(MutationDelete), params.IdempotencyKey, params.RequestDigest)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, false, fmt.Errorf("request Audit deletion: %w", err)
	}
	if replay, found, replayErr := s.lookupAuditReplay(
		ctx, params.OwnerID, string(MutationDelete), params.IdempotencyKey, params.RequestDigest,
	); replayErr != nil || found {
		return replay, false, replayErr
	}
	if _, getErr := s.Get(ctx, params.OwnerID, params.AuditID); errors.Is(getErr, ErrNotFound) {
		return Audit{}, false, ErrNotFound
	} else if getErr != nil {
		return Audit{}, false, getErr
	}
	return Audit{}, false, ErrPrecondition
}
