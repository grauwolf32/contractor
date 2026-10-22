package auditservice

import (
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

const (
	MaxReviewRationale = 64 << 10
	defaultReviewTTL   = 7 * 24 * time.Hour
	maximumReviewTTL   = 30 * 24 * time.Hour
)

type Finding struct {
	Fingerprint       *auditdomain.FindingFingerprint `json:"fingerprint,omitempty"`
	FindingID         string                          `json:"findingId"`
	AuditID           string                          `json:"auditId"`
	State             auditstore.FindingState         `json:"state"`
	RejectionReason   *string                         `json:"rejectionReason,omitempty"`
	DuplicateTargetID *string                         `json:"duplicateTargetId,omitempty"`
	FirstProposal     findingintake.Receipt           `json:"firstProposal"`
	CurrentAssessment *auditstore.FindingAssessment   `json:"currentAssessment,omitempty"`
	AnalystDecision   *auditstore.ReviewDecision      `json:"analystDecision,omitempty"`
	AnalystVerdict    *auditstore.AnalystVerdict      `json:"analystVerdict,omitempty"`
	AnalystSeverity   *auditstore.FindingSeverity     `json:"analystSeverity,omitempty"`
	Revision          uint64                          `json:"revision"`
	CreatedAt         time.Time                       `json:"createdAt"`
	UpdatedAt         time.Time                       `json:"updatedAt"`
}

type CreateFindingReviewParams struct {
	OwnerID          string
	AuditID          string
	FindingID        string
	ExpectedRevision uint64
	RequestID        string
	ExpiresAt        *time.Time
	IdempotencyKey   string
	RequestDigest    string
}

type FindingReviewResult struct {
	Request  auditstore.ReviewRequest
	Replayed bool
}

type DecideFindingParams struct {
	OwnerID                 string
	AuditID                 string
	RequestID               string
	ExpectedRequestRevision uint64
	DecisionID              string
	Verdict                 auditstore.AnalystVerdict
	Severity                *auditstore.FindingSeverity
	Rationale               string
	DuplicateTargetID       *string
	IdempotencyKey          string
	RequestDigest           string
}

type FindingDecisionResult struct {
	Finding  Finding                   `json:"finding"`
	Request  auditstore.ReviewRequest  `json:"request"`
	Decision auditstore.ReviewDecision `json:"decision"`
	Replayed bool                      `json:"replayed"`
}

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

type FindingProvenance struct {
	RecordID        string                           `json:"recordId"`
	Kind            auditstore.FindingProvenanceKind `json:"kind"`
	ReceiptID       string                           `json:"receiptId"`
	Relation        string                           `json:"relation,omitempty"`
	Proposal        auditstore.ExactArtifact         `json:"proposal"`
	Origin          findingintake.Origin             `json:"origin"`
	Assessment      *auditstore.FindingAssessment    `json:"assessment,omitempty"`
	Attempt         *auditstore.FindingAttempt       `json:"attempt,omitempty"`
	SupportsCurrent bool                             `json:"supportsCurrentAssessment"`
	CreatedAt       time.Time                        `json:"createdAt"`
}
