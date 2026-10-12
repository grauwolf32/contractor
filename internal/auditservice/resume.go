package auditservice

import (
	"context"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/auditstore/reviewstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

func validateDeadlineSeconds(seconds *int) error {
	if seconds != nil && (*seconds < 0 || *seconds > auditprofile.MaxAuditDeadlineSeconds) {
		return fmt.Errorf("%w: Audit time limit must be between 0 (unlimited) and %d seconds", ErrInvalid, auditprofile.MaxAuditDeadlineSeconds)
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
	if replay, found, err := s.owners().LookupMutationReplay(ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest); err != nil || found {
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
	owners, reads := ownerstore.NewTx(tx), querystore.New(tx)
	// Serialize retries before any mutable dependency checks.
	if err := owners.LockMutation(ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey); err != nil {
		return MutationResult{}, err
	}
	if replay, found, err := owners.LookupMutationReplay(ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest); err != nil || found {
		return MutationResult{Audit: replay, Replayed: found}, err
	}
	audit, err := reads.Get(ctx, params.OwnerID, params.AuditID)
	if err != nil {
		return MutationResult{}, err
	}
	if _, err := projectstore.NewPostgresStore(tx).LockActiveAuditProject(ctx, params.OwnerID, audit.ProjectID); err != nil {
		return MutationResult{}, err
	}
	if err := owners.LockAuditForResume(ctx, params.OwnerID, params.AuditID); err != nil {
		return MutationResult{}, err
	}
	audit, err = reads.Get(ctx, params.OwnerID, params.AuditID)
	if err != nil {
		return MutationResult{}, err
	}
	if audit.Revision != params.ExpectedRevision || audit.State != auditstore.AuditPaused || len(audit.BaselineSnapshot) == 0 {
		return MutationResult{}, auditstore.ErrPrecondition
	}
	candidate, err := owners.HasReportReview(ctx, audit.AuditID)
	if err != nil {
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
	// Expired human decisions are never silently extended by resuming an
	// Audit. Each needs a fresh decision for the same exact task before
	// another submission; the PostgreSQL clock judges expiry, and the fresh
	// requests expire with the new deadline.
	revision, err := reviewstore.NewTx(tx).RenewItemReviewsForResume(ctx, auditstore.ResumeRenewalParams{
		OwnerID: params.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision, DeadlineAt: deadline,
	})
	if err != nil {
		return MutationResult{}, err
	}
	resumed, inserted, err := owners.Resume(ctx, auditstore.ResumeParams{
		OwnerID: params.OwnerID, AuditID: audit.AuditID, ExpectedRevision: revision,
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
	profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil {
		return nil, err
	}
	deadline := now.Add(timeDurationSeconds(profile.Execution.DeadlineSeconds))
	return &deadline, nil
}
