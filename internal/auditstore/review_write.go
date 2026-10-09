package auditstore

import (
	"context"
	"encoding/json"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// FindingReviewRequestParams pins the subject already validated and locked by
// the service. These primitives use its transaction and do not change lock order.
type FindingReviewRequestParams struct {
	RequestID, AuditID, FindingID, SubjectDigest string
	SubjectRevision                              uint64
	RequestedActions                             json.RawMessage
	ExpiresAt                                    *time.Time
	DefaultTTLSeconds                            int64
	IdempotencyKey, RequestDigest                string
}

func (s *PostgresStore) InsertFindingReviewRequest(ctx context.Context, p FindingReviewRequestParams) error {
	_, err := s.db.Exec(ctx, insertFindingReviewRequestSQL, p.RequestID, p.AuditID,
		p.FindingID, p.SubjectRevision, p.SubjectDigest, p.RequestedActions,
		p.ExpiresAt, p.IdempotencyKey, p.RequestDigest, p.DefaultTTLSeconds)
	return reviewWriteError(err)
}

// ReviewDecisionWriteParams records either a finding verdict or an item/report
// action. Finding-only fields stay NULL for the latter.
type ReviewDecisionWriteParams struct {
	DecisionID, RequestID, AuditID, ActorID, Action         string
	FindingID, Verdict, Severity, DuplicateTargetID         *string
	Rationale, SubjectDigest, IdempotencyKey, RequestDigest string
	SubjectRevision                                         uint64
}

func (s *PostgresStore) RecordReviewDecision(ctx context.Context, p ReviewDecisionWriteParams) error {
	_, err := s.db.Exec(ctx, recordReviewDecisionSQL, p.DecisionID, p.RequestID,
		p.AuditID, p.FindingID, p.ActorID, p.Action, p.Verdict, p.Severity,
		p.Rationale, p.DuplicateTargetID, p.SubjectRevision, p.SubjectDigest,
		p.IdempotencyKey, p.RequestDigest)
	return reviewWriteError(err)
}

func (s *PostgresStore) ProjectFindingDecision(ctx context.Context, findingID, state string,
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
func (s *PostgresStore) ExpireReviewAtDatabaseTime(ctx context.Context, requestID string) (bool, error) {
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
func (s *PostgresStore) ExpireFindingReviews(ctx context.Context, auditID, findingID string) error {
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
	events := make([]ReviewEventParams, 0)
	for rows.Next() {
		var id string
		var revision uint64
		if err := rows.Scan(&id, &revision); err != nil {
			return err
		}
		events = append(events, ReviewEventParams{AuditID: auditID, Kind: "review.expired",
			EntityID: id, EntityRevision: &revision, Summary: map[string]any{
				"subjectKind": "finding", "subjectId": findingID, "findingId": findingID, "kind": "finding-triage",
			}})
	}
	if err := rows.Err(); err != nil {
		return err
	}
	return s.AppendReviewEvents(ctx, events)
}

func reviewWriteError(err error) error {
	if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateUniqueViolation {
		return ErrConflict
	}
	return err
}
