package pgstore

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// The steps of an owner's review decision inside the caller's transaction.
// Each lock lasts until the caller commits, so these steps exist only on the
// transaction-bound ReviewTxStore (PER-R18).

// LockActionReviews serializes the item action and report decisions of one
// Audit until the caller commits.
func (s *ReviewTxStore) LockActionReviews(ctx context.Context, auditID string) error {
	return persistencepostgres.LockAdvisory(ctx, s.tx, persistencepostgres.NewAdvisoryKey("audit-action-review", auditID))
}

// FindReviewDecisionReplay reads the decision an Audit stored under an
// idempotency key.
func (s *ReviewTxStore) FindReviewDecisionReplay(
	ctx context.Context, auditID, idempotencyKey string,
) (auditstore.ReviewDecisionReplay, bool, error) {
	var result auditstore.ReviewDecisionReplay
	err := s.tx.QueryRow(ctx, `
SELECT decision_id, request_id, request_digest
  FROM audit_review_decisions
 WHERE audit_id = $1 AND idempotency_key = $2`, auditID, idempotencyKey).Scan(
		&result.DecisionID, &result.RequestID, &result.RequestDigest,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ReviewDecisionReplay{}, false, nil
	}
	if err != nil {
		return auditstore.ReviewDecisionReplay{}, false, err
	}
	return result, true, nil
}

// LockAuditForDecision locks an owner's Audit row ahead of its review
// request, in the Controller's audit, review, item lock order.
func (s *ReviewTxStore) LockAuditForDecision(ctx context.Context, ownerID, auditID string) error {
	var lockedAuditID string
	err := s.tx.QueryRow(ctx, `
SELECT audit_id FROM audits
 WHERE owner_id=$1 AND audit_id=$2 FOR UPDATE`, ownerID, auditID).Scan(&lockedAuditID)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ErrNotFound
	}
	return err
}

// LockActionReview locks an item action or report review request of an
// owner's Audit with the Audit.
func (s *ReviewTxStore) LockActionReview(
	ctx context.Context, ownerID, auditID, requestID string,
) (auditstore.LockedActionReview, error) {
	var result auditstore.LockedActionReview
	err := s.tx.QueryRow(ctx, lockActionReviewSQL, ownerID, auditID, requestID).Scan(
		&result.AuditState, &result.AuditRevision, &result.SubjectKind, &result.SubjectID, &result.Kind,
		&result.SubjectRevision, &result.SubjectDigest, &result.RequestedActions, &result.State,
		&result.ExpiresAt, &result.Revision,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.LockedActionReview{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.LockedActionReview{}, err
	}
	return result, nil
}
