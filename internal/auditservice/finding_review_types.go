package auditservice

import (
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

const (
	MaxFindingPageSize = 200
	MaxReviewRationale = 64 << 10
	defaultReviewTTL   = 7 * 24 * time.Hour
	maximumReviewTTL   = 30 * 24 * time.Hour
	FindingReviewKind  = "finding-triage"
)

type FindingState string

const (
	FindingProposed      FindingState = "proposed"
	FindingConfirmed     FindingState = "confirmed"
	FindingRejected      FindingState = "rejected"
	FindingDuplicate     FindingState = "duplicate"
	FindingNeedsEvidence FindingState = "needs-evidence"
)

func (s FindingState) Valid() bool {
	switch s {
	case FindingProposed, FindingConfirmed, FindingRejected, FindingDuplicate, FindingNeedsEvidence:
		return true
	default:
		return false
	}
}

type AnalystVerdict string

const (
	VerdictTruePositive  AnalystVerdict = "true_positive"
	VerdictFalsePositive AnalystVerdict = "false_positive"
	VerdictDuplicate     AnalystVerdict = "duplicate"
	VerdictReopen        AnalystVerdict = "reopen"
	VerdictNeedsEvidence AnalystVerdict = "needs_evidence"
)

func (v AnalystVerdict) Valid() bool {
	switch v {
	case VerdictTruePositive, VerdictFalsePositive, VerdictDuplicate, VerdictReopen, VerdictNeedsEvidence:
		return true
	default:
		return false
	}
}

type FindingSeverity string

const (
	SeverityInformational FindingSeverity = "informational"
	SeverityLow           FindingSeverity = "low"
	SeverityMedium        FindingSeverity = "medium"
	SeverityHigh          FindingSeverity = "high"
	SeverityCritical      FindingSeverity = "critical"
)

func (s FindingSeverity) Valid() bool {
	switch s {
	case SeverityInformational, SeverityLow, SeverityMedium, SeverityHigh, SeverityCritical:
		return true
	default:
		return false
	}
}

type ReviewState string

const (
	ReviewPending ReviewState = "pending"
	ReviewDecided ReviewState = "decided"
	ReviewExpired ReviewState = "expired"
)

func (s ReviewState) Valid() bool {
	return s == ReviewPending || s == ReviewDecided || s == ReviewExpired
}

type FindingAssessment struct {
	AssessmentID        string                    `json:"assessmentId"`
	SemanticAssessment  string                    `json:"semanticAssessment"`
	Result              auditstore.ExactArtifact  `json:"result"`
	ReceiptID           string                    `json:"receiptId"`
	ItemID              *string                   `json:"itemId,omitempty"`
	ExecutionItemID     *string                   `json:"executionItemId,omitempty"`
	CollectionReceiptID *string                   `json:"collectionReceiptId,omitempty"`
	DirectVerification  bool                      `json:"directVerification"`
	Contract            *auditstore.ExactArtifact `json:"contract,omitempty"`
	AcceptedAt          time.Time                 `json:"acceptedAt"`
}

type ReviewDecision struct {
	DecisionID        string           `json:"decisionId"`
	RequestID         string           `json:"requestId"`
	AuditID           string           `json:"auditId"`
	FindingID         string           `json:"findingId"`
	ActorID           string           `json:"actorId"`
	Verdict           AnalystVerdict   `json:"verdict"`
	Severity          *FindingSeverity `json:"severity,omitempty"`
	Rationale         string           `json:"rationale"`
	DuplicateTargetID *string          `json:"duplicateTargetId,omitempty"`
	SubjectRevision   uint64           `json:"subjectRevision"`
	SubjectDigest     string           `json:"subjectDigest"`
	CreatedAt         time.Time        `json:"createdAt"`
}

type ReviewRequest struct {
	RequestID        string           `json:"requestId"`
	AuditID          string           `json:"auditId"`
	FindingID        string           `json:"findingId"`
	Kind             string           `json:"kind"`
	SubjectRevision  uint64           `json:"subjectRevision"`
	SubjectDigest    string           `json:"subjectDigest"`
	RequestedActions []AnalystVerdict `json:"requestedActions"`
	State            ReviewState      `json:"state"`
	ExpiresAt        *time.Time       `json:"expiresAt,omitempty"`
	Revision         uint64           `json:"revision"`
	Decision         *ReviewDecision  `json:"decision,omitempty"`
	CreatedAt        time.Time        `json:"createdAt"`
	UpdatedAt        time.Time        `json:"updatedAt"`
}

type Finding struct {
	FindingID         string                `json:"findingId"`
	AuditID           string                `json:"auditId"`
	State             FindingState          `json:"state"`
	RejectionReason   *string               `json:"rejectionReason,omitempty"`
	DuplicateTargetID *string               `json:"duplicateTargetId,omitempty"`
	FirstProposal     findingintake.Receipt `json:"firstProposal"`
	CurrentAssessment *FindingAssessment    `json:"currentAssessment,omitempty"`
	AnalystDecision   *ReviewDecision       `json:"analystDecision,omitempty"`
	AnalystVerdict    *AnalystVerdict       `json:"analystVerdict,omitempty"`
	AnalystSeverity   *FindingSeverity      `json:"analystSeverity,omitempty"`
	Revision          uint64                `json:"revision"`
	CreatedAt         time.Time             `json:"createdAt"`
	UpdatedAt         time.Time             `json:"updatedAt"`
}

type FindingListParams struct {
	OwnerID        string
	AuditID        string
	State          *FindingState
	Verdict        *AnalystVerdict
	Unreviewed     bool
	Severity       *FindingSeverity
	AfterCreatedAt *time.Time
	AfterFindingID string
	Limit          int
}

type ReviewListParams struct {
	OwnerID        string
	AuditID        string
	FindingID      *string
	State          *ReviewState
	AfterCreatedAt *time.Time
	AfterRequestID string
	Limit          int
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
	Request  ReviewRequest
	Replayed bool
}

type DecideFindingParams struct {
	OwnerID                 string
	AuditID                 string
	RequestID               string
	ExpectedRequestRevision uint64
	DecisionID              string
	Verdict                 AnalystVerdict
	Severity                *FindingSeverity
	Rationale               string
	DuplicateTargetID       *string
	IdempotencyKey          string
	RequestDigest           string
}

type FindingDecisionResult struct {
	Finding  Finding        `json:"finding"`
	Request  ReviewRequest  `json:"request"`
	Decision ReviewDecision `json:"decision"`
	Replayed bool           `json:"replayed"`
}

type FindingProvenanceKind string

const (
	ProvenanceSourceProposal FindingProvenanceKind = "source-proposal"
	ProvenanceCheckAttempt   FindingProvenanceKind = "check-attempt"
	ProvenanceDirect         FindingProvenanceKind = "direct-verification"
)

type FindingProvenance struct {
	RecordID        string                   `json:"recordId"`
	Kind            FindingProvenanceKind    `json:"kind"`
	ReceiptID       string                   `json:"receiptId"`
	Relation        string                   `json:"relation,omitempty"`
	Proposal        auditstore.ExactArtifact `json:"proposal"`
	Origin          findingintake.Origin     `json:"origin"`
	Assessment      *FindingAssessment       `json:"assessment,omitempty"`
	SupportsCurrent bool                     `json:"supportsCurrentAssessment"`
	CreatedAt       time.Time                `json:"createdAt"`
}

type ProvenanceListParams struct {
	OwnerID        string
	AuditID        string
	FindingID      string
	AfterCreatedAt *time.Time
	AfterRecordID  string
	Limit          int
}
