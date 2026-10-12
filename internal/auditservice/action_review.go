package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/reviewstore"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// DecideActionReview applies non-finding human authority through the same
// immutable review ledger as finding triage. The exact subject is revalidated
// while both the request and its target are locked; execution intent creation
// performs an independent final authorization check.
func (s *Service) DecideActionReview(
	ctx context.Context, params DecideActionReviewParams,
) (ActionReviewDecisionResult, error) {
	if err := validateActionDecision(params); err != nil {
		return ActionReviewDecisionResult{}, err
	}
	var decisionID, requestID string
	var replayed, expired bool
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		reviews := reviewstore.NewTx(tx)
		if err := reviews.LockActionReviews(ctx, params.AuditID); err != nil {
			return err
		}
		replay, found, err := reviews.FindReviewDecisionReplay(ctx, params.AuditID, params.IdempotencyKey)
		if err != nil {
			return err
		}
		if found {
			decisionID, requestID = replay.DecisionID, replay.RequestID
			if replay.RequestDigest != params.RequestDigest || replay.RequestID != params.RequestID {
				return fmt.Errorf("%w: review decision key", contracts.ErrIdempotencyConflict)
			}
			replayed = true
			return nil
		}
		// Match the controller's audit -> review -> item lock order. A decision
		// must not hold an item review while waiting for its Audit row.
		if err := reviews.LockAuditForDecision(ctx, params.OwnerID, params.AuditID); err != nil {
			return err
		}
		locked, err := reviews.LockActionReview(ctx, params.OwnerID, params.AuditID, params.RequestID)
		if err != nil {
			return err
		}
		auditState, auditRevision := locked.AuditState, locked.AuditRevision
		subjectKind, subjectID, kind := locked.SubjectKind, locked.SubjectID, locked.Kind
		subjectRevision, subjectDigest := locked.SubjectRevision, locked.SubjectDigest
		if locked.State != auditstore.ReviewPending || locked.Revision != int64(params.ExpectedRequestRevision) ||
			subjectRevision < 1 || auditState == auditstore.AuditDeleting ||
			auditState == auditstore.AuditFinalizing {
			return auditstore.ErrPrecondition
		}
		var requested []auditstore.ReviewAction
		if json.Unmarshal(locked.RequestedActions, &requested) != nil || !containsReviewAction(requested, params.Action) {
			return auditstore.ErrPrecondition
		}
		if locked.ExpiresAt != nil {
			if expired, err = reviews.ExpireReviewAtDatabaseTime(ctx, params.RequestID); err != nil {
				return err
			}
		}
		if expired {
			return appendAuditReviewEvent(ctx, tx, params.AuditID, "review.expired",
				params.RequestID, map[string]any{
					"subjectKind": subjectKind,
					"subjectId":   subjectID,
					"kind":        kind,
				})
		}

		switch subjectKind {
		case auditstore.ReviewSubjectItemAction:
			if auditState != auditstore.AuditActive && auditState != auditstore.AuditWaitingReview &&
				auditState != auditstore.AuditPaused {
				return auditstore.ErrPrecondition
			}
			if err := reviews.ApplyItemReviewDecision(
				ctx, params.AuditID, subjectID, kind, subjectDigest,
				string(params.Action), params.Rationale,
			); err != nil {
				return err
			}
		case auditstore.ReviewSubjectReport:
			if auditState != auditstore.AuditWaitingReview || kind != auditstore.ReportAcceptanceReviewKind {
				return auditstore.ErrPrecondition
			}
			if err := validateAndApplyReportDecision(
				ctx, reviews, params.AuditID, auditRevision, params.RequestID, subjectID,
				subjectRevision, subjectDigest, params.Action,
			); err != nil {
				return err
			}
		default:
			return auditstore.ErrPrecondition
		}

		decisionID, requestID = params.DecisionID, params.RequestID
		if err := reviews.RecordReviewDecision(ctx, auditstore.ReviewDecisionWriteParams{
			DecisionID: decisionID, RequestID: requestID, AuditID: params.AuditID, ActorID: params.OwnerID,
			Action: string(params.Action), Rationale: params.Rationale, SubjectRevision: uint64(subjectRevision),
			SubjectDigest: subjectDigest, IdempotencyKey: params.IdempotencyKey, RequestDigest: params.RequestDigest,
		}); err != nil {
			return err
		}
		if subjectKind == auditstore.ReviewSubjectItemAction && auditState == auditstore.AuditWaitingReview {
			if _, _, err := reviews.ActivateAfterItemReview(
				ctx, params.AuditID, auditRevision,
			); err != nil {
				return err
			}
		}
		return appendAuditReviewEvent(ctx, tx, params.AuditID, "review.decided", decisionID,
			map[string]any{
				"subjectKind": subjectKind, "subjectId": subjectID,
				"kind": kind, "action": params.Action,
			})
	})
	if err != nil {
		return ActionReviewDecisionResult{}, err
	}
	if expired {
		return ActionReviewDecisionResult{}, auditstore.ErrPrecondition
	}
	request, err := s.GetReview(ctx, params.OwnerID, params.AuditID, requestID)
	if err != nil {
		return ActionReviewDecisionResult{}, err
	}
	if request.Decision == nil || request.Decision.DecisionID != decisionID {
		return ActionReviewDecisionResult{}, errors.New("stored Audit action decision is missing")
	}
	return ActionReviewDecisionResult{
		Request: request, Decision: *request.Decision, Replayed: replayed,
	}, nil
}

// validateAndApplyReportDecision applies the owner's decision on the exact
// frozen report candidate through the store transitions, which record the
// same Audit events as the Controller's report commit and terminal paths.
func validateAndApplyReportDecision(
	ctx context.Context, reviews *reviewstore.TxStore, auditID string, auditRevision uint64,
	requestID, subjectID string, subjectRevision int64, subjectDigest string,
	action auditstore.ReviewAction,
) error {
	if subjectID != auditID || subjectRevision < 1 {
		return auditstore.ErrPrecondition
	}
	decision := auditstore.ReportDecisionParams{
		AuditID: auditID, ExpectedAuditRevision: auditRevision, RequestID: requestID,
		SubjectRevision: uint64(subjectRevision), SubjectDigest: subjectDigest,
	}
	var err error
	switch action {
	case auditstore.ReviewApprove:
		_, err = reviews.AcceptReportCandidate(ctx, decision)
	case auditstore.ReviewReject:
		_, err = reviews.RejectReportCandidate(ctx, decision)
	default:
		return auditstore.ErrInvalid
	}
	return err
}

func containsReviewAction(values []auditstore.ReviewAction, target auditstore.ReviewAction) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func validateActionDecision(params DecideActionReviewParams) error {
	if !auditstore.ValidReviewIdentity(params.OwnerID, 256) || !auditstore.ValidReviewIdentity(params.AuditID, 256) ||
		!auditstore.ValidReviewIdentity(params.RequestID, 256) || !auditstore.ValidReviewIdentity(params.DecisionID, 256) ||
		params.ExpectedRequestRevision == 0 || !params.Action.Valid() ||
		!validIdempotencyKey(params.IdempotencyKey) || !contentdigest.Valid(params.RequestDigest) ||
		!validRationale(params.Rationale) {
		return auditstore.ErrInvalid
	}
	return nil
}
