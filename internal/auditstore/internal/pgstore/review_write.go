package pgstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// RecordReviewDecision appends the decision and marks its request decided
// together. It returns ErrPrecondition and writes nothing unless request
// p.RequestID of Audit p.AuditID is pending. Callers lock and validate the
// request before; the store keeps the guarantee on its own as well.
func (s *ReviewStore) RecordReviewDecision(ctx context.Context, p auditstore.ReviewDecisionWriteParams) error {
	tag, err := s.db.Exec(ctx, recordReviewDecisionSQL, p.DecisionID, p.RequestID,
		p.AuditID, p.ActorID, p.Action, p.Rationale, p.SubjectRevision, p.SubjectDigest,
		p.IdempotencyKey, p.RequestDigest)
	if err != nil {
		return reviewWriteError(err)
	}
	if tag.RowsAffected() != 1 {
		return auditstore.ErrPrecondition
	}
	return nil
}

// ExpireReviewAtDatabaseTime uses the same clock as execution authorization.
// The caller holds the request lock and appends the event in its transaction.
func (s *ReviewStore) ExpireReviewAtDatabaseTime(ctx context.Context, requestID string) (bool, error) {
	tag, err := s.db.Exec(ctx, `
UPDATE audit_review_requests
   SET state='expired', revision=revision+1,
       updated_at=GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE request_id=$1 AND state='pending'
   AND expires_at IS NOT NULL AND expires_at <= clock_timestamp()`, requestID)
	return tag.RowsAffected() == 1, err
}

// ExpireFindingReviews records every expired request before a replacement is
// inserted. The caller holds the Audit and finding locks in the same transaction.

// reviewWriteError classifies a refused review write. A CHECK violation is a
// value the service should have rejected; nothing was written and repeating
// the request cannot succeed.
func reviewWriteError(err error) error {
	switch persistencepostgres.Classify(err) {
	case persistencepostgres.ClassDuplicate:
		return auditstore.ErrConflict
	case persistencepostgres.ClassInvalid:
		return invalidf("review write violates a stored constraint: %v", err)
	}
	return err
}
