package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *PostgresStore) Transition(
	ctx context.Context,
	params TransitionParams,
) (Audit, bool, error) {
	if err := validateTransition(params); err != nil {
		return Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, "audit.transition", params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	var reasonCode, reasonMessage *string
	if params.Reason != nil {
		reasonCode, reasonMessage = &params.Reason.Code, &params.Reason.Message
	}
	response, _ := json.Marshal(map[string]string{"auditId": params.AuditID})
	audit, err := scanAudit(s.db.QueryRow(ctx, `
WITH active_project_gate AS MATERIALIZED (
    SELECT CASE WHEN $5 = 'active'
           THEN contractor_require_active_audit_project(project_id, owner_id)
           END
      FROM audits
     WHERE audit_id = $2 AND owner_id = $1
), changed AS (
    UPDATE audits AS audit
       SET state = $5,
           revision = revision + 1,
           dispatch_state = CASE
               WHEN $5 IN ('finalizing', 'cancelling', 'completed', 'cancelled', 'failed', 'deleting')
                   THEN 'closed'
               ELSE dispatch_state
           END,
           stop_reason_code = $6,
           stop_reason_message = $7,
           finished_at = CASE WHEN $5 IN ('completed', 'cancelled', 'failed')
                              THEN clock_timestamp() ELSE NULL END,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond'),
           next_event_sequence = next_event_sequence + 1
      FROM active_project_gate
     WHERE audit.owner_id = $1 AND audit.audit_id = $2
       AND audit.revision = $3 AND audit.state = $4
    RETURNING audit.*
), idempotency_row AS (
    INSERT INTO audit_idempotency (
        owner_id, operation, idempotency_key, request_digest,
        audit_id, resource_id, response_snapshot
    )
    SELECT $1, 'audit.transition', $8, $9, audit_id, audit_id, $10::jsonb
      FROM changed
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT audit_id, next_event_sequence - 1, 'audit.state_changed', audit_id, revision,
           jsonb_build_object('from', $4::text, 'to', $5::text)
      FROM changed
)
SELECT `+prefixedAuditColumns("changed")+` FROM changed`,
		params.OwnerID, params.AuditID, params.ExpectedRevision,
		string(params.ExpectedState), string(params.TargetState), reasonCode, reasonMessage,
		params.IdempotencyKey, params.RequestDigest, response,
	))
	if err == nil {
		return audit, true, nil
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return Audit{}, false, ErrProjectDeleting
	}
	if persistencepostgres.SQLState(err) == "23505" {
		return s.replayAudit(ctx, params.OwnerID, "audit.transition", params.IdempotencyKey, params.RequestDigest)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, false, fmt.Errorf("transition Audit: %w", err)
	}
	if replay, found, replayErr := s.lookupAuditReplay(
		ctx, params.OwnerID, "audit.transition", params.IdempotencyKey, params.RequestDigest,
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

func (s *PostgresStore) TransitionClaimed(
	ctx context.Context,
	params ClaimedTransitionParams,
) (Audit, error) {
	if err := validateClaimedTransition(params); err != nil {
		return Audit{}, err
	}
	var reasonCode, reasonMessage *string
	if params.Reason != nil {
		reasonCode, reasonMessage = &params.Reason.Code, &params.Reason.Message
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, `
	WITH live_claim AS MATERIALIZED (
	    SELECT claim.audit_id
	      FROM audit_controller_claims AS claim
	     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
	       AND claim.expires_at > clock_timestamp()
	     FOR UPDATE OF claim
	), claim_gate AS MATERIALIZED (
	    SELECT audit.audit_id,
	           CASE WHEN $6 = 'active'
	                THEN contractor_require_active_audit_project(audit.project_id, audit.owner_id)
	           END
	      FROM audits AS audit
	      JOIN live_claim USING (audit_id)
	     WHERE audit.audit_id = $1
	     FOR UPDATE OF audit
	), changed AS (
    UPDATE audits AS audit
       SET state = $6,
           revision = audit.revision + 1,
           dispatch_state = CASE
               WHEN $6 IN ('finalizing', 'cancelling', 'completed', 'cancelled', 'failed', 'deleting')
                   THEN 'closed'
               ELSE audit.dispatch_state
           END,
           stop_reason_code = $7,
           stop_reason_message = $8,
           finished_at = CASE WHEN $6 IN ('completed', 'cancelled', 'failed')
                              THEN clock_timestamp() ELSE NULL END,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond'),
           next_event_sequence = audit.next_event_sequence + 1
      FROM claim_gate
     WHERE audit.audit_id = claim_gate.audit_id
       AND audit.revision = $4 AND audit.state = $5
    RETURNING audit.*
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT audit_id, next_event_sequence - 1, 'audit.state_changed', audit_id, revision,
           jsonb_build_object('from', $5::text, 'to', $6::text)
      FROM changed
)
SELECT `+prefixedAuditColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExpectedRevision, string(params.ExpectedState), string(params.TargetState),
		reasonCode, reasonMessage,
	))
	if err == nil {
		return audit, nil
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return Audit{}, ErrProjectDeleting
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, fmt.Errorf("transition claimed Audit: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Audit{}, liveErr
	} else if !live {
		return Audit{}, ErrClaimLost
	}
	existing, getErr := s.getAuditTrusted(ctx, params.Claim.AuditID)
	if getErr != nil {
		return Audit{}, getErr
	}
	if existing.State == params.TargetState && existing.Revision == params.ExpectedRevision+1 {
		return existing, nil
	}
	return Audit{}, ErrPrecondition
}

func (s *PostgresStore) getAuditTrusted(ctx context.Context, auditID string) (Audit, error) {
	audit, err := scanAudit(s.db.QueryRow(ctx, `
SELECT `+auditColumns+` FROM audits WHERE audit_id = $1`, auditID))
	if errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, ErrNotFound
	}
	if err != nil {
		return Audit{}, fmt.Errorf("read trusted Audit: %w", err)
	}
	return audit, nil
}
