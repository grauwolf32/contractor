package auditstore

import (
	"encoding/json"
	"time"
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

// ReviewDecisionWriteParams records either a finding verdict or an item/report
// action. Finding-only fields stay NULL for the latter.
type ReviewDecisionWriteParams struct {
	DecisionID, RequestID, AuditID, ActorID, Action         string
	FindingID, Verdict, Severity, DuplicateTargetID         *string
	Rationale, SubjectDigest, IdempotencyKey, RequestDigest string
	SubjectRevision                                         uint64
}
