package auditservice

import (
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

const (
	MaxReviewRationale = 64 << 10
	defaultReviewTTL   = 7 * 24 * time.Hour
	maximumReviewTTL   = 30 * 24 * time.Hour
)

type DecideActionReviewParams struct {
	OwnerID                 string
	AuditID                 string
	RequestID               string
	ExpectedRequestRevision uint64
	DecisionID              string
	Action                  auditstore.ReviewAction
	Rationale               string
	IdempotencyKey          string
	RequestDigest           string
}

type ActionReviewDecisionResult struct {
	Request  auditstore.ReviewRequest  `json:"request"`
	Decision auditstore.ReviewDecision `json:"decision"`
	Replayed bool                      `json:"replayed"`
}
