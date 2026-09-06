package auditservice

import (
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

const (
	MaxFindingPageSize         = 200
	MaxReviewRationale         = 64 << 10
	defaultReviewTTL           = 7 * 24 * time.Hour
	maximumReviewTTL           = 30 * 24 * time.Hour
	FindingReviewKind          = "finding-triage"
	ActiveCheckReviewKind      = "active-check-approval"
	ApplicabilityReviewKind    = "requirement-applicability"
	ReportAcceptanceReviewKind = "report-acceptance"
)

func (kind ReviewSubjectKind) Valid() bool {
	return kind == ReviewSubjectFinding || kind == ReviewSubjectItemAction ||
		kind == ReviewSubjectReport
}

type ReviewSubjectKind string

const (
	ReviewSubjectFinding    ReviewSubjectKind = "finding"
	ReviewSubjectItemAction ReviewSubjectKind = "audit-item-action"
	ReviewSubjectReport     ReviewSubjectKind = "audit-report"
)

type ReviewAction string

const (
	ReviewApprove       ReviewAction = "approve"
	ReviewReject        ReviewAction = "reject"
	ReviewNotApplicable ReviewAction = "not_applicable"
)

func (action ReviewAction) Valid() bool {
	return action == ReviewApprove || action == ReviewReject || action == ReviewNotApplicable
}

// ReviewRequestedAction is the closed public union used by the shared review
// ledger. Finding triage and approval decisions deliberately keep distinct
// typed request bodies even though their immutable request history is listed
// through one endpoint.
type ReviewRequestedAction string

func (action ReviewRequestedAction) Valid() bool {
	if ReviewAction(action).Valid() {
		return true
	}
	return AnalystVerdict(action).Valid()
}

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
	FindingID         string           `json:"findingId,omitempty"`
	Action            ReviewAction     `json:"action,omitempty"`
	ActorID           string           `json:"actorId"`
	Verdict           AnalystVerdict   `json:"verdict,omitempty"`
	Severity          *FindingSeverity `json:"severity,omitempty"`
	Rationale         string           `json:"rationale"`
	DuplicateTargetID *string          `json:"duplicateTargetId,omitempty"`
	SubjectRevision   uint64           `json:"subjectRevision"`
	SubjectDigest     string           `json:"subjectDigest"`
	CreatedAt         time.Time        `json:"createdAt"`
}

type ReviewRequest struct {
	RequestID        string                  `json:"requestId"`
	AuditID          string                  `json:"auditId"`
	FindingID        string                  `json:"findingId,omitempty"`
	SubjectKind      ReviewSubjectKind       `json:"subjectKind"`
	SubjectID        string                  `json:"subjectId"`
	Kind             string                  `json:"kind"`
	SubjectRevision  uint64                  `json:"subjectRevision"`
	SubjectDigest    string                  `json:"subjectDigest"`
	RequestedActions []ReviewRequestedAction `json:"requestedActions"`
	State            ReviewState             `json:"state"`
	ExpiresAt        *time.Time              `json:"expiresAt,omitempty"`
	Revision         uint64                  `json:"revision"`
	Decision         *ReviewDecision         `json:"decision,omitempty"`
	CreatedAt        time.Time               `json:"createdAt"`
	UpdatedAt        time.Time               `json:"updatedAt"`
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

type DecideActionReviewParams struct {
	OwnerID                 string
	AuditID                 string
	RequestID               string
	ExpectedRequestRevision uint64
	DecisionID              string
	Action                  ReviewAction
	Rationale               string
	IdempotencyKey          string
	RequestDigest           string
}

type ActionReviewDecisionResult struct {
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

func (kind FindingProvenanceKind) Valid() bool {
	return kind == ProvenanceSourceProposal || kind == ProvenanceCheckAttempt ||
		kind == ProvenanceDirect
}

type FindingProvenance struct {
	RecordID        string                   `json:"recordId"`
	Kind            FindingProvenanceKind    `json:"kind"`
	ReceiptID       string                   `json:"receiptId"`
	Relation        string                   `json:"relation,omitempty"`
	Proposal        auditstore.ExactArtifact `json:"proposal"`
	Origin          findingintake.Origin     `json:"origin"`
	Assessment      *FindingAssessment       `json:"assessment,omitempty"`
	Attempt         *FindingAttempt          `json:"attempt,omitempty"`
	SupportsCurrent bool                     `json:"supportsCurrentAssessment"`
	CreatedAt       time.Time                `json:"createdAt"`
}

// FindingAttempt preserves the trusted verification execution independently
// of whether that attempt produced the assessment currently selected for the
// finding. It therefore keeps failed, invalid, inconclusive and superseded
// attempts visible after their ordinary child Run is deleted.
type FindingAttempt struct {
	ExecutionID           string                            `json:"executionId"`
	ExecutionItemID       string                            `json:"executionItemId"`
	ItemID                string                            `json:"itemId"`
	ItemAttempt           int                               `json:"itemAttempt"`
	Role                  auditstore.ExecutionRole          `json:"role"`
	WorkflowRole          string                            `json:"workflowRole"`
	State                 auditstore.ItemState              `json:"state"`
	CollectionDisposition *auditstore.CollectionDisposition `json:"collectionDisposition,omitempty"`
	TerminalOutcome       *auditstore.TerminalOutcome       `json:"terminalOutcome,omitempty"`
	RunID                 *string                           `json:"runId,omitempty"`
	RunDeleted            bool                              `json:"runDeleted"`
	RunProvenance         *auditstore.RunProvenance         `json:"runProvenance,omitempty"`
	Task                  auditstore.ExactArtifact          `json:"task"`
	ItemOrigin            auditstore.ItemOrigin             `json:"itemOrigin"`
	Result                *auditstore.ExactArtifact         `json:"result,omitempty"`
	CreatedAt             time.Time                         `json:"createdAt"`
	CollectedAt           *time.Time                        `json:"collectedAt,omitempty"`
}

type ProvenanceListParams struct {
	OwnerID        string
	AuditID        string
	FindingID      string
	AfterCreatedAt *time.Time
	AfterRecordID  string
	Limit          int
}
