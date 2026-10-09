package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contentdigest"
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
		if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`,
			"audit-action-review:"+params.AuditID); err != nil {
			return err
		}
		var storedDigest string
		err := tx.QueryRow(ctx, `
SELECT decision_id, request_id, request_digest
  FROM audit_review_decisions
 WHERE audit_id = $1 AND idempotency_key = $2`,
			params.AuditID, params.IdempotencyKey,
		).Scan(&decisionID, &requestID, &storedDigest)
		if err == nil {
			if storedDigest != params.RequestDigest || requestID != params.RequestID {
				return auditstore.ErrConflict
			}
			replayed = true
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}
		// Match the controller's audit -> review -> item lock order. A decision
		// must not hold an item review while waiting for its Audit row.
		var lockedAuditID string
		if err := tx.QueryRow(ctx, `
SELECT audit_id FROM audits
 WHERE owner_id=$1 AND audit_id=$2 FOR UPDATE`, params.OwnerID, params.AuditID).
			Scan(&lockedAuditID); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return auditstore.ErrNotFound
			}
			return err
		}

		var auditState auditstore.AuditState
		var subjectKind ReviewSubjectKind
		var subjectID, kind, subjectDigest string
		var auditRevision uint64
		var subjectRevision, requestRevision int64
		var requestState ReviewState
		var expiresAt *time.Time
		var requestedJSON []byte
		err = tx.QueryRow(ctx, `
SELECT audit.state, audit.revision, request.subject_kind, request.subject_id, request.kind,
       request.subject_revision, request.subject_digest,
       request.requested_actions, request.state, request.expires_at,
       request.revision
  FROM audit_review_requests AS request
  JOIN audits AS audit USING (audit_id)
 WHERE audit.owner_id = $1 AND request.audit_id = $2
   AND request.request_id = $3
 FOR UPDATE OF audit, request`,
			params.OwnerID, params.AuditID, params.RequestID,
		).Scan(&auditState, &auditRevision, &subjectKind, &subjectID, &kind, &subjectRevision,
			&subjectDigest, &requestedJSON, &requestState, &expiresAt, &requestRevision)
		if errors.Is(err, pgx.ErrNoRows) {
			return auditstore.ErrNotFound
		}
		if err != nil {
			return err
		}
		if requestState != ReviewPending || requestRevision != int64(params.ExpectedRequestRevision) ||
			subjectRevision < 1 || auditState == auditstore.AuditDeleting ||
			auditState == auditstore.AuditFinalizing {
			return auditstore.ErrPrecondition
		}
		var requested []ReviewAction
		if json.Unmarshal(requestedJSON, &requested) != nil || !containsReviewAction(requested, params.Action) {
			return auditstore.ErrPrecondition
		}
		if expiresAt != nil {
			if expired, err = auditstore.NewPostgresStore(tx).ExpireReviewAtDatabaseTime(ctx, params.RequestID); err != nil {
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
		case ReviewSubjectItemAction:
			if auditState != auditstore.AuditActive && auditState != auditstore.AuditWaitingReview &&
				auditState != auditstore.AuditPaused {
				return auditstore.ErrPrecondition
			}
			if err := auditstore.NewPostgresStore(tx).ApplyItemReviewDecision(
				ctx, params.AuditID, subjectID, kind, subjectDigest,
				string(params.Action), params.Rationale,
			); err != nil {
				return err
			}
		case ReviewSubjectReport:
			if auditState != auditstore.AuditWaitingReview || kind != ReportAcceptanceReviewKind {
				return auditstore.ErrPrecondition
			}
			if err := validateAndApplyReportDecision(
				ctx, tx, params.AuditID, auditRevision, params.RequestID, subjectID,
				subjectRevision, subjectDigest, params.Action,
			); err != nil {
				return err
			}
		default:
			return auditstore.ErrPrecondition
		}

		decisionID, requestID = params.DecisionID, params.RequestID
		if err := auditstore.NewPostgresStore(tx).RecordReviewDecision(ctx, auditstore.ReviewDecisionWriteParams{
			DecisionID: decisionID, RequestID: requestID, AuditID: params.AuditID, ActorID: params.OwnerID,
			Action: string(params.Action), Rationale: params.Rationale, SubjectRevision: uint64(subjectRevision),
			SubjectDigest: subjectDigest, IdempotencyKey: params.IdempotencyKey, RequestDigest: params.RequestDigest,
		}); err != nil {
			return err
		}
		if subjectKind == ReviewSubjectItemAction && auditState == auditstore.AuditWaitingReview {
			if _, _, err := auditstore.NewPostgresStore(tx).ActivateAfterItemReview(
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
	ctx context.Context, tx pgx.Tx, auditID string, auditRevision uint64,
	requestID, subjectID string, subjectRevision int64, subjectDigest string,
	action ReviewAction,
) error {
	if subjectID != auditID || subjectRevision < 1 {
		return auditstore.ErrPrecondition
	}
	decision := auditstore.ReportDecisionParams{
		AuditID: auditID, ExpectedAuditRevision: auditRevision, RequestID: requestID,
		SubjectRevision: uint64(subjectRevision), SubjectDigest: subjectDigest,
	}
	store := auditstore.NewPostgresStore(tx)
	var err error
	switch action {
	case ReviewApprove:
		_, err = store.AcceptReportCandidate(ctx, decision)
	case ReviewReject:
		_, err = store.RejectReportCandidate(ctx, decision)
	default:
		return auditstore.ErrInvalid
	}
	return err
}

func containsReviewAction(values []ReviewAction, target ReviewAction) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func validateActionDecision(params DecideActionReviewParams) error {
	if !validReviewIdentity(params.OwnerID, 256) || !validReviewIdentity(params.AuditID, 256) ||
		!validReviewIdentity(params.RequestID, 256) || !validReviewIdentity(params.DecisionID, 256) ||
		params.ExpectedRequestRevision == 0 || !params.Action.Valid() ||
		!validReviewIdentity(params.IdempotencyKey, 128) || !contentdigest.Valid(params.RequestDigest) ||
		!validRationale(params.Rationale) {
		return auditstore.ErrInvalid
	}
	return nil
}
