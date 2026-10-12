package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// Transition applies an owner pause or cancellation; validateTransition admits
// no other owner target. Leaving waiting_review closes a pending report review
// in the same revision step.
func (s *OwnerStore) Transition(
	ctx context.Context,
	params auditstore.TransitionParams,
) (auditstore.Audit, bool, error) {
	if err := validateTransition(params); err != nil {
		return auditstore.Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	var reasonCode, reasonMessage *string
	if params.Reason != nil {
		reasonCode, reasonMessage = &params.Reason.Code, &params.Reason.Message
	}
	response, _ := json.Marshal(map[string]string{"auditId": params.AuditID})
	audit, err := scanAudit(persistencepostgres.SavepointQueryRow(ctx, s.db, transitionSQL+prefixedAuditColumns("changed")+` FROM changed`,
		params.OwnerID, params.AuditID, params.ExpectedRevision,
		string(params.ExpectedState), string(params.TargetState), reasonCode, reasonMessage,
		params.IdempotencyKey, params.RequestDigest, response,
	))
	if err == nil {
		return audit, true, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDuplicate {
		return s.replayAudit(ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, false, fmt.Errorf("transition Audit: %w", err)
	}
	if replay, found, replayErr := s.lookupAuditReplay(
		ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest,
	); replayErr != nil || found {
		return replay, false, replayErr
	}
	if _, getErr := s.get(ctx, params.OwnerID, params.AuditID); errors.Is(getErr, auditstore.ErrNotFound) {
		return auditstore.Audit{}, false, auditstore.ErrNotFound
	} else if getErr != nil {
		return auditstore.Audit{}, false, getErr
	}
	return auditstore.Audit{}, false, auditstore.ErrPrecondition
}

func (s *ClaimStore) TransitionClaimed(
	ctx context.Context,
	params auditstore.ClaimedTransitionParams,
) (auditstore.Audit, error) {
	if err := ValidateClaimedTransition(params); err != nil {
		return auditstore.Audit{}, err
	}
	var reasonCode, reasonMessage *string
	if params.Reason != nil {
		reasonCode, reasonMessage = &params.Reason.Code, &params.Reason.Message
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, transitionClaimedSQL+prefixedAuditColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExpectedRevision, string(params.ExpectedState), string(params.TargetState),
		reasonCode, reasonMessage,
	))
	if err == nil {
		return audit, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting {
		return auditstore.Audit{}, auditstore.ErrProjectDeleting
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, fmt.Errorf("transition claimed Audit: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return auditstore.Audit{}, liveErr
	} else if !live {
		return auditstore.Audit{}, auditstore.ErrClaimLost
	}
	existing, getErr := s.getAuditTrusted(ctx, params.Claim.AuditID)
	if getErr != nil {
		return auditstore.Audit{}, getErr
	}
	if existing.State == params.TargetState && existing.Revision == params.ExpectedRevision+1 {
		return existing, nil
	}
	return auditstore.Audit{}, auditstore.ErrPrecondition
}

// Resume reopens dispatch for a paused Audit with a fresh hold and deadline.
// It records the owner idempotency row and an audit.resumed event carrying
// the cleared stop reason, behind the same active-Project gate.
func (s *OwnerStore) Resume(ctx context.Context, params auditstore.ResumeParams) (auditstore.Audit, bool, error) {
	if err := validateResume(params); err != nil {
		return auditstore.Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	deadline, _ := json.Marshal(params.DeadlineAt)
	response, _ := json.Marshal(map[string]string{"auditId": params.AuditID})
	audit, err := scanAudit(persistencepostgres.SavepointQueryRow(ctx, s.db, resumeSQL+prefixedAuditColumns("changed")+` FROM changed`,
		params.OwnerID, params.AuditID, params.ExpectedRevision, params.DeadlineAt, deadline,
		params.IdempotencyKey, params.RequestDigest, response,
	))
	if err == nil {
		return audit, true, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting {
		return auditstore.Audit{}, false, auditstore.ErrProjectDeleting
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDuplicate {
		return s.replayAudit(ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, false, fmt.Errorf("resume Audit: %w", err)
	}
	if replay, found, replayErr := s.lookupAuditReplay(
		ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest,
	); replayErr != nil || found {
		return replay, false, replayErr
	}
	if _, getErr := s.get(ctx, params.OwnerID, params.AuditID); getErr != nil {
		return auditstore.Audit{}, false, getErr
	}
	return auditstore.Audit{}, false, auditstore.ErrPrecondition
}

// transitionTrusted applies a service-authorized transition with the same
// transition table, active-Project gate and state event as other transitions.
func (s *ReviewStore) transitionTrusted(
	ctx context.Context,
	params auditstore.TrustedTransitionParams,
) (auditstore.Audit, error) {
	if err := validateTrustedTransition(params); err != nil {
		return auditstore.Audit{}, err
	}
	var reasonCode, reasonMessage *string
	if params.Reason != nil {
		reasonCode, reasonMessage = &params.Reason.Code, &params.Reason.Message
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, transitionTrustedSQL+prefixedAuditColumns("changed")+` FROM changed`,
		params.AuditID, params.ExpectedRevision,
		string(params.ExpectedState), string(params.TargetState), reasonCode, reasonMessage,
	))
	if err == nil {
		return audit, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting {
		return auditstore.Audit{}, auditstore.ErrProjectDeleting
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, fmt.Errorf("transition trusted Audit: %w", err)
	}
	if _, getErr := s.getAuditTrusted(ctx, params.AuditID); getErr != nil {
		return auditstore.Audit{}, getErr
	}
	return auditstore.Audit{}, auditstore.ErrPrecondition
}

// ActivateAfterItemReview returns a waiting_review Audit to active after an
// item decision unless every unsettled item still awaits review, so an
// approved item becomes dispatchable without waiting for its siblings.
func (s *ReviewStore) ActivateAfterItemReview(
	ctx context.Context, auditID string, expectedRevision uint64,
) (auditstore.Audit, bool, error) {
	onlyAwaiting, err := s.onlyAwaitingReview(ctx, auditID)
	if err != nil || onlyAwaiting {
		return auditstore.Audit{}, false, err
	}
	audit, err := s.transitionTrusted(ctx, auditstore.TrustedTransitionParams{
		AuditID: auditID, ExpectedRevision: expectedRevision,
		ExpectedState: auditstore.AuditWaitingReview, TargetState: auditstore.AuditActive,
	})
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	return audit, true, nil
}

func (s *store) getAuditTrusted(ctx context.Context, auditID string) (auditstore.Audit, error) {
	audit, err := scanAudit(s.db.QueryRow(ctx, `
SELECT `+prefixedAuditColumns("audits")+` FROM audits WHERE audit_id = $1`, auditID))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.Audit{}, fmt.Errorf("read trusted Audit: %w", err)
	}
	return audit, nil
}
