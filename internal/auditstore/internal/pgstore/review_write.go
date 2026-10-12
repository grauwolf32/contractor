package pgstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func (s *ReviewStore) InsertFindingReviewRequest(ctx context.Context, p auditstore.FindingReviewRequestParams) error {
	_, err := s.db.Exec(ctx, insertFindingReviewRequestSQL, p.RequestID, p.AuditID,
		p.FindingID, p.SubjectRevision, p.SubjectDigest, p.RequestedActions,
		p.ExpiresAt, p.IdempotencyKey, p.RequestDigest, p.DefaultTTLSeconds)
	return reviewWriteError(err)
}

func (s *ReviewStore) RecordReviewDecision(ctx context.Context, p auditstore.ReviewDecisionWriteParams) error {
	_, err := s.db.Exec(ctx, recordReviewDecisionSQL, p.DecisionID, p.RequestID,
		p.AuditID, p.FindingID, p.ActorID, p.Action, p.Verdict, p.Severity,
		p.Rationale, p.DuplicateTargetID, p.SubjectRevision, p.SubjectDigest,
		p.IdempotencyKey, p.RequestDigest)
	return reviewWriteError(err)
}

func (s *ReviewStore) ProjectFindingDecision(ctx context.Context, findingID, state string,
	rejection, duplicate, decision *string,
) error {
	_, err := s.db.Exec(ctx, `
UPDATE audit_findings
   SET state=$2, rejection_reason=$3, duplicate_target_id=$4,
       current_decision_id=$5, revision=revision+1,
       updated_at=GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE finding_id=$1`, findingID, state, rejection, duplicate, decision)
	return err
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
func (s *ReviewStore) ExpireFindingReviews(ctx context.Context, auditID, findingID string) error {
	rows, err := s.db.Query(ctx, `
UPDATE audit_review_requests
   SET state='expired', revision=revision+1,
       updated_at=GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id=$1 AND finding_id=$2 AND state='pending'
   AND expires_at IS NOT NULL AND expires_at <= clock_timestamp()
RETURNING request_id, revision`, auditID, findingID)
	if err != nil {
		return err
	}
	defer rows.Close()
	events := make([]auditstore.ReviewEventParams, 0)
	for rows.Next() {
		var id string
		var revision uint64
		if err := rows.Scan(&id, &revision); err != nil {
			return err
		}
		events = append(events, auditstore.ReviewEventParams{AuditID: auditID, Kind: "review.expired",
			EntityID: id, EntityRevision: &revision, Summary: map[string]any{
				"subjectKind": "finding", "subjectId": findingID, "findingId": findingID, "kind": "finding-triage",
			}})
	}
	if err := rows.Err(); err != nil {
		return err
	}
	return s.appendReviewEvents(ctx, events)
}

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
