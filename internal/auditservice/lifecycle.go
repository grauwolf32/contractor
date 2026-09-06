package auditservice

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

type lifecycleAction string

const (
	lifecyclePause  lifecycleAction = "pause"
	lifecycleResume lifecycleAction = "resume"
	lifecycleCancel lifecycleAction = "cancel"
)

func (s *Service) Pause(ctx context.Context, params MutationParams) (MutationResult, error) {
	return s.transition(ctx, params, lifecyclePause)
}

func (s *Service) Resume(ctx context.Context, params MutationParams) (MutationResult, error) {
	return s.transition(ctx, params, lifecycleResume)
}

func (s *Service) Cancel(ctx context.Context, params MutationParams) (MutationResult, error) {
	return s.transition(ctx, params, lifecycleCancel)
}

func (s *Service) Delete(ctx context.Context, params MutationParams) (MutationResult, error) {
	if err := validateMutationParams(params); err != nil {
		return MutationResult{}, err
	}
	store := auditstore.NewPostgresStore(s.pool)
	if replay, found, err := store.LookupMutationReplay(
		ctx, params.OwnerID, auditstore.MutationDelete, params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return MutationResult{Audit: replay, Replayed: found}, err
	}
	audit, changed, err := store.RequestDelete(ctx, auditstore.DeleteParams{
		OwnerID: params.OwnerID, AuditID: params.AuditID,
		ExpectedRevision: params.ExpectedRevision, IdempotencyKey: params.IdempotencyKey,
		RequestDigest: params.RequestDigest,
	})
	return MutationResult{Audit: audit, Replayed: !changed}, err
}

func (s *Service) transition(
	ctx context.Context, params MutationParams, action lifecycleAction,
) (MutationResult, error) {
	if err := validateMutationParams(params); err != nil {
		return MutationResult{}, err
	}
	store := auditstore.NewPostgresStore(s.pool)
	if replay, found, err := store.LookupMutationReplay(
		ctx, params.OwnerID, auditstore.MutationTransition, params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return MutationResult{Audit: replay, Replayed: found}, err
	}
	audit, err := store.Get(ctx, params.OwnerID, params.AuditID)
	if err != nil {
		return MutationResult{}, err
	}
	// A report awaiting exact owner acceptance is an immutable projection
	// snapshot. Pausing it would make resume re-enter ordinary reconciliation
	// with a candidate created from an older Audit revision. Cancellation and
	// deletion remain available, while accept/reject uses the review endpoint.
	if action == lifecyclePause && audit.State == auditstore.AuditWaitingReview {
		if _, candidateErr := store.GetReportCandidate(ctx, audit.AuditID); candidateErr == nil {
			return MutationResult{}, auditstore.ErrPrecondition
		} else if !errors.Is(candidateErr, auditstore.ErrNotFound) {
			return MutationResult{}, candidateErr
		}
	}
	target, reason, err := lifecycleTarget(action, audit.State)
	if err != nil {
		return MutationResult{}, err
	}
	changed, inserted, err := store.Transition(ctx, auditstore.TransitionParams{
		OwnerID: params.OwnerID, AuditID: params.AuditID,
		ExpectedRevision: params.ExpectedRevision, ExpectedState: audit.State, TargetState: target,
		Reason: reason, IdempotencyKey: params.IdempotencyKey, RequestDigest: params.RequestDigest,
	})
	return MutationResult{Audit: changed, Replayed: !inserted}, err
}

func lifecycleTarget(
	action lifecycleAction, state auditstore.AuditState,
) (auditstore.AuditState, *auditstore.StopReason, error) {
	switch action {
	case lifecyclePause:
		if state == auditstore.AuditActive || state == auditstore.AuditWaitingReview {
			return auditstore.AuditPaused, nil, nil
		}
	case lifecycleResume:
		if state == auditstore.AuditPaused {
			return auditstore.AuditActive, nil, nil
		}
	case lifecycleCancel:
		if state == auditstore.AuditActive || state == auditstore.AuditWaitingReview ||
			state == auditstore.AuditPaused || state == auditstore.AuditFinalizing {
			return auditstore.AuditCancelling, &auditstore.StopReason{
				Code: "cancel_requested", Message: "Audit cancellation was requested by its owner.",
			}, nil
		}
	}
	return "", nil, auditstore.ErrPrecondition
}

func validateMutationParams(params MutationParams) error {
	if params.OwnerID == "" || params.AuditID == "" || params.ExpectedRevision == 0 ||
		params.IdempotencyKey == "" || params.RequestDigest == "" {
		return fmt.Errorf("%w: Audit mutation is incomplete", ErrInvalid)
	}
	return nil
}
