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
           revision = candidate.revision + 1,
           next_event_sequence = candidate.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), candidate.updated_at + interval '1 microsecond')
      FROM candidate
     WHERE audit.audit_id = candidate.audit_id AND candidate.state <> 'deleting'
    RETURNING audit.*
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
    SELECT audit_id, next_event_sequence - 1, 'audit.delete_requested', audit_id, revision,
           jsonb_build_object('state', state)
      FROM changed
)
SELECT `+prefixedAuditColumns("selected")+` FROM selected`,
		params.OwnerID, params.AuditID, params.ExpectedRevision,
		params.IdempotencyKey, params.RequestDigest, response,
	))
	if err == nil {
		return audit, true, nil
	}
	if persistencepostgres.SQLState(err) == "23505" {
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
