package auditservice

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

func validateDeadlineSeconds(seconds *int) error {
	if seconds != nil && (*seconds < 0 || *seconds > config.MaxAuditDeadlineSeconds) {
		return fmt.Errorf("%w: Audit time limit must be between 0 (unlimited) and %d seconds", ErrInvalid, config.MaxAuditDeadlineSeconds)
	}
	return nil
}

// Resume continues a paused Audit with its pinned baseline and accepted results.
// Terminal Audits cannot be reopened, including historical deadline closures.
func (s *Service) Resume(ctx context.Context, params MutationParams) (MutationResult, error) {
	if err := validateMutationParams(params); err != nil {
		return MutationResult{}, err
	}
	if err := validateDeadlineSeconds(params.DeadlineSeconds); err != nil {
		return MutationResult{}, err
	}
	store := auditstore.NewPostgresStore(s.pool)
	if replay, found, err := store.LookupMutationReplay(ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest); err != nil || found {
		return MutationResult{Audit: replay, Replayed: found}, err
	}
	var result MutationResult
	err := s.credentialGuard.WithRunCreation(ctx, func() error {
		return persistencepostgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			result, err = s.resumeInTransaction(ctx, tx, params)
			return err
		})
	})
	return result, err
}

func (s *Service) resumeInTransaction(ctx context.Context, tx pgx.Tx, params MutationParams) (MutationResult, error) {
	store := auditstore.NewPostgresStore(tx)
	// Serialize retries before any mutable dependency checks.
	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, "audit-resume:"+params.OwnerID+":"+params.IdempotencyKey); err != nil {
		return MutationResult{}, err
	}
	if replay, found, err := store.LookupMutationReplay(ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest); err != nil || found {
		return MutationResult{Audit: replay, Replayed: found}, err
	}
	audit, err := store.Get(ctx, params.OwnerID, params.AuditID)
	if err != nil {
		return MutationResult{}, err
	}
	if _, err := projectstore.NewPostgresStore(tx).LockActiveAuditProject(ctx, params.OwnerID, audit.ProjectID); err != nil {
		return MutationResult{}, err
	}
	if _, err := tx.Exec(ctx, `SELECT audit_id FROM audits WHERE audit_id=$1 AND owner_id=$2 FOR UPDATE`, params.AuditID, params.OwnerID); err != nil {
		return MutationResult{}, err
	}
	audit, err = store.Get(ctx, params.OwnerID, params.AuditID)
	if err != nil {
		return MutationResult{}, err
	}
	if audit.Revision != params.ExpectedRevision || audit.State != auditstore.AuditPaused || audit.CurrentRoundID == nil {
		return MutationResult{}, auditstore.ErrPrecondition
	}
	var candidate bool
	if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audit_report_candidates WHERE audit_id=$1)`, audit.AuditID).Scan(&candidate); err != nil {
		return MutationResult{}, err
	}
	if candidate {
		return MutationResult{}, auditstore.ErrPrecondition
	}
	now := s.now().UTC()
	deadline, err := resumeDeadline(audit, params.DeadlineSeconds, now)
	if err != nil {
		return MutationResult{}, err
	}
	if err := renewExpiredItemReviews(ctx, tx, audit, deadline, now); err != nil {
		return MutationResult{}, err
	}
	if _, err := tx.Exec(ctx, `UPDATE audits SET state='active', dispatch_state='open', hold_state='held', deadline_at=$2, paused_at=NULL, finished_at=NULL, stop_reason_code=NULL, stop_reason_message=NULL, revision=revision+1, next_event_sequence=next_event_sequence+1, updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond') WHERE audit_id=$1`, audit.AuditID, deadline); err != nil {
		return MutationResult{}, err
	}
	summary, _ := json.Marshal(map[string]any{"from": audit.State, "to": "active", "previousStopReason": audit.StopReason, "deadlineAt": deadline})
	if _, err := tx.Exec(ctx, `INSERT INTO audit_events(audit_id,sequence_number,kind,entity_id,entity_revision,summary) SELECT audit_id,next_event_sequence-1,'audit.resumed',audit_id,revision,$2::jsonb FROM audits WHERE audit_id=$1`, audit.AuditID, summary); err != nil {
		return MutationResult{}, err
	}
	response, _ := json.Marshal(map[string]string{"auditId": audit.AuditID})
	if _, err := tx.Exec(ctx, `INSERT INTO audit_idempotency(owner_id,operation,idempotency_key,request_digest,audit_id,resource_id,response_snapshot) VALUES($1,'audit.transition',$2,$3,$4,$4,$5::jsonb)`, params.OwnerID, params.IdempotencyKey, params.RequestDigest, audit.AuditID, response); err != nil {
		return MutationResult{}, err
	}
	updated, err := store.Get(ctx, params.OwnerID, audit.AuditID)
	return MutationResult{Audit: updated}, err
}

func resumeDeadline(audit auditstore.Audit, seconds *int, now time.Time) (*time.Time, error) {
	if seconds != nil {
		if *seconds == 0 {
			return nil, nil
		}
		deadline := now.Add(timeDurationSeconds(*seconds))
		return &deadline, nil
	}
	if audit.State == auditstore.AuditPaused && audit.DeadlineAt == nil {
		return nil, nil
	}
	if audit.State == auditstore.AuditPaused && audit.PausedAt != nil && audit.DeadlineAt != nil && audit.DeadlineAt.After(*audit.PausedAt) {
		deadline := now.Add(audit.DeadlineAt.Sub(*audit.PausedAt))
		return &deadline, nil
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil {
		return nil, err
	}
	deadline := now.Add(timeDurationSeconds(profile.Execution.DeadlineSeconds))
	return &deadline, nil
}

// Expired human decisions are never silently extended by resuming an Audit.
// Require a fresh decision for the same exact task before another submission.
func renewExpiredItemReviews(ctx context.Context, tx pgx.Tx, audit auditstore.Audit, deadline *time.Time, now time.Time) error {
	_, err := tx.Exec(ctx, `UPDATE audit_review_requests SET state='expired',revision=revision+1,updated_at=clock_timestamp() WHERE audit_id=$1 AND subject_kind='audit-item-action' AND state='pending' AND expires_at <= $2`, audit.AuditID, now)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `WITH needs_review AS (
        UPDATE audit_items AS item SET state='awaiting_review',updated_at=clock_timestamp()
        WHERE item.audit_id=$1 AND item.state IN ('ready','awaiting_review') AND item.approval_kind <> 'none'
          AND NOT EXISTS (SELECT 1 FROM audit_review_requests r WHERE r.audit_id=item.audit_id AND r.subject_id=item.item_id AND r.subject_kind='audit-item-action' AND r.subject_digest=item.approval_subject_digest AND (r.expires_at IS NULL OR r.expires_at>$2) AND (r.state='pending' OR (r.state='decided' AND EXISTS(SELECT 1 FROM audit_review_decisions d WHERE d.request_id=r.request_id AND d.action='approve'))))
        RETURNING item.*
    ) INSERT INTO audit_review_requests(request_id,audit_id,finding_id,subject_kind,subject_id,kind,subject_revision,subject_digest,requested_actions,state,expires_at,idempotency_key,request_digest)
      SELECT 'review-resume-'||$3::text||'-'||item.item_id,item.audit_id,NULL,'audit-item-action',item.item_id,item.approval_kind,1,item.approval_subject_digest,
        CASE WHEN item.approval_kind='requirement-applicability' THEN '["approve","reject","not_applicable"]'::jsonb ELSE '["approve","reject"]'::jsonb END,'pending',$4,'resume:'||$3::text||':'||item.item_id,item.approval_subject_digest FROM needs_review item`, audit.AuditID, now, fmt.Sprint(audit.Revision), deadline)
	return err
}
