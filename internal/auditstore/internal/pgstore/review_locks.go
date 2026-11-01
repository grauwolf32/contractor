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

// LockFindingReviews serializes the finding triage requests and decisions of
// one Audit until the caller commits.

// LockActionReviews serializes the item action and report decisions of one
// Audit until the caller commits.
func (s *ReviewTxStore) LockActionReviews(ctx context.Context, auditID string) error {
	return persistencepostgres.LockAdvisory(ctx, s.tx, persistencepostgres.NewAdvisoryKey("audit-action-review", auditID))
}

// FindReviewRequestReplay reads the review request an Audit stored under an
// idempotency key and the digest of the request that created it.
func (s *ReviewTxStore) FindReviewRequestReplay(
	ctx context.Context, auditID, idempotencyKey string,
) (requestID, requestDigest string, found bool, err error) {
	err = s.tx.QueryRow(ctx, `
SELECT request_id, request_digest
  FROM audit_review_requests
 WHERE audit_id = $1 AND idempotency_key = $2`, auditID, idempotencyKey).Scan(&requestID, &requestDigest)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", "", false, nil
	}
	if err != nil {
		return "", "", false, err
	}
	return requestID, requestDigest, true, nil
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

// LockFinding locks one finding of an owner's Audit with the Audit and reads
// the finding and the Audit's state.

// PendingReportReview reports whether an Audit has a pending report review.
// The caller has locked the Audit row; report proposal takes the same lock,
// so the answer holds until the caller commits.
func (s *ReviewTxStore) PendingReportReview(ctx context.Context, auditID string) (bool, error) {
	var pending bool
	err := s.tx.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1
      FROM audit_review_requests AS request
     WHERE request.audit_id = $1 AND request.subject_kind = 'audit-report'
       AND request.state = 'pending'
)`, auditID).Scan(&pending)
	return pending, err
}

// PendingFindingReview reads the oldest pending triage request of a finding
// for its exact subject revision and digest.

// LockFindingReview locks a finding triage request of an owner's Audit with
// its finding and Audit. The request's requested actions and decision are
// not read.

// DuplicateTarget reports whether finding targetID of an Audit exists and
// whether marking findingID a duplicate of it would close a cycle of
// duplicate targets.

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
