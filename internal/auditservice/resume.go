package auditservice

import (
	"context"
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
	if err := renewExpiredItemReviews(ctx, tx, audit, deadline); err != nil {
		return MutationResult{}, err
	}
	resumed, inserted, err := store.Resume(ctx, auditstore.ResumeParams{
		OwnerID: params.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		DeadlineAt: deadline, IdempotencyKey: params.IdempotencyKey, RequestDigest: params.RequestDigest,
	})
	return MutationResult{Audit: resumed, Replayed: !inserted}, err
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
// Expiry is judged by the PostgreSQL clock, like execution authorization.
func renewExpiredItemReviews(ctx context.Context, tx pgx.Tx, audit auditstore.Audit, deadline *time.Time) error {
	_, err := tx.Exec(ctx, `UPDATE audit_review_requests SET state='expired',revision=revision+1,updated_at=clock_timestamp() WHERE audit_id=$1 AND subject_kind='audit-item-action' AND state='pending' AND expires_at <= clock_timestamp()`, audit.AuditID)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `WITH needs_review AS (
        UPDATE audit_items AS item SET state='awaiting_review',updated_at=clock_timestamp()
        WHERE item.audit_id=$1 AND item.state IN ('ready','awaiting_review') AND item.approval_kind <> 'none'
          AND NOT EXISTS (SELECT 1 FROM audit_review_requests r WHERE r.audit_id=item.audit_id AND r.subject_id=item.item_id AND r.subject_kind='audit-item-action' AND r.subject_digest=item.approval_subject_digest AND (r.expires_at IS NULL OR r.expires_at>clock_timestamp()) AND (r.state='pending' OR (r.state='decided' AND EXISTS(SELECT 1 FROM audit_review_decisions d WHERE d.request_id=r.request_id AND d.action='approve'))))
        RETURNING item.*
    ) INSERT INTO audit_review_requests(request_id,audit_id,finding_id,subject_kind,subject_id,kind,subject_revision,subject_digest,requested_actions,state,expires_at,idempotency_key,request_digest)
      SELECT 'review-resume-'||$2::text||'-'||item.item_id,item.audit_id,NULL,'audit-item-action',item.item_id,item.approval_kind,1,item.approval_subject_digest,
        CASE WHEN item.approval_kind='requirement-applicability' THEN '["approve","reject","not_applicable"]'::jsonb ELSE '["approve","reject"]'::jsonb END,'pending',$3,'resume:'||$2::text||':'||item.item_id,item.approval_subject_digest FROM needs_review item`, audit.AuditID, fmt.Sprint(audit.Revision), deadline)
	return err
}
