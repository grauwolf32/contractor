package auditstore

import (
	"encoding/json"
	"strings"
	"time"
	"unicode/utf8"
)

// The review journal: finding triage, item action and report acceptance
// requests, their decisions, and the findings they decide.

// ValidReviewIdentity is the identity rule of review requests, review
// decisions, findings and the owners and Audits that hold them: non-blank
// UTF-8 without NUL, at most maximum bytes. The review writes and the review
// and finding reads apply this one definition.
func ValidReviewIdentity(value string, maximum int) bool {
	return strings.TrimSpace(value) != "" && utf8.ValidString(value) &&
		!strings.ContainsRune(value, 0) && len([]byte(value)) <= maximum
}

// An item action review's kind is its item's approval kind
// (ItemApprovalKind); the other review kinds are these.
const (
	FindingReviewKind          = "finding-triage"
	ReportAcceptanceReviewKind = "report-acceptance"
)

type ReviewSubjectKind string

const (
	ReviewSubjectFinding    ReviewSubjectKind = "finding"
	ReviewSubjectItemAction ReviewSubjectKind = "audit-item-action"
	ReviewSubjectReport     ReviewSubjectKind = "audit-report"
)

func (kind ReviewSubjectKind) Valid() bool {
	return kind == ReviewSubjectFinding || kind == ReviewSubjectItemAction ||
		kind == ReviewSubjectReport
}

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

// FindingRecord is one stored finding: its triage state and the current
// assessment and analyst decision it points to.
type FindingRecord struct {
	FindingID           string
	AuditID             string
	FirstReceiptID      string
	State               FindingState
	RejectionReason     *string
	DuplicateTargetID   *string
	CurrentAssessmentID *string
	CurrentDecisionID   *string
	Revision            uint64
	CreatedAt           time.Time
	UpdatedAt           time.Time
}

type FindingAssessment struct {
	AssessmentID        string         `json:"assessmentId"`
	SemanticAssessment  string         `json:"semanticAssessment"`
	Result              ExactArtifact  `json:"result"`
	ReceiptID           string         `json:"receiptId"`
	ItemID              *string        `json:"itemId,omitempty"`
	ExecutionItemID     *string        `json:"executionItemId,omitempty"`
	CollectionReceiptID *string        `json:"collectionReceiptId,omitempty"`
	DirectVerification  bool           `json:"directVerification"`
	Contract            *ExactArtifact `json:"contract,omitempty"`
	AcceptedAt          time.Time      `json:"acceptedAt"`
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

// ReviewDecisionReplay is the decision an Audit already stored under an
// owner's idempotency key. FindingID is empty for an action decision.
type ReviewDecisionReplay struct {
	DecisionID    string
	RequestID     string
	FindingID     string
	RequestDigest string
}

// LockedFindingReview is a finding triage request and its finding, locked
// with their Audit for a decision, and the Audit's state.
type LockedFindingReview struct {
	AuditState AuditState
	Request    ReviewRequest
	Finding    FindingRecord
}

// LockedActionReview is an item action or report acceptance request, locked
// with its Audit for a decision, and the Audit's state and revision.
// RequestedActions is the stored JSON array of requested actions.
type LockedActionReview struct {
	AuditState       AuditState
	AuditRevision    uint64
	SubjectKind      ReviewSubjectKind
	SubjectID        string
	Kind             string
	SubjectRevision  int64
	SubjectDigest    string
	RequestedActions json.RawMessage
	State            ReviewState
	ExpiresAt        *time.Time
	Revision         int64
}

type FindingListParams struct {
	AuditRevision  *uint64
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
	AuditRevision  *uint64
	OwnerID        string
	AuditID        string
	FindingID      *string
	State          *ReviewState
	AfterCreatedAt *time.Time
	AfterRequestID string
	Limit          int
}

// Owner lists are live, newest-first keyset pages across active Projects.
// AuditID and AuditRevision are intentionally absent from these contracts.
type OwnerFindingListParams struct {
	OwnerID         string
	State           *FindingState
	Verdict         *AnalystVerdict
	Unreviewed      bool
	Severity        *FindingSeverity
	AuditStates     []AuditState
	BeforeCreatedAt *time.Time
	BeforeFindingID string
	Limit           int
}

type OwnerReviewListParams struct {
	OwnerID         string
	State           *ReviewState
	AuditStates     []AuditState
	BeforeCreatedAt *time.Time
	BeforeRequestID string
	Limit           int
}

// PageBasis is the Audit revision, statement time and whole-filter total a
// finding or review page is read against.
type PageBasis struct {
	AuditRevision uint64    `json:"auditRevision"`
	AsOf          time.Time `json:"asOf"`
	Total         int       `json:"total"`
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

// FindingProvenanceRecord is one stored record that produced a finding: a
// source proposal, a check attempt or a direct verification.
type FindingProvenanceRecord struct {
	RecordID   string
	Kind       FindingProvenanceKind
	ReceiptID  string
	Relation   string
	Proposal   ExactArtifact
	Assessment *FindingAssessment
	Attempt    *FindingAttempt
	CreatedAt  time.Time
}

// FindingAttempt preserves the trusted verification execution independently
// of whether that attempt produced the assessment currently selected for the
// finding. It therefore keeps failed, invalid, inconclusive and superseded
// attempts visible after their ordinary child Run is deleted.
type FindingAttempt struct {
	ExecutionID           string                 `json:"executionId"`
	ExecutionItemID       string                 `json:"executionItemId"`
	ItemID                string                 `json:"itemId"`
	ItemAttempt           int                    `json:"itemAttempt"`
	Role                  ExecutionRole          `json:"role"`
	WorkflowRole          string                 `json:"workflowRole"`
	State                 ItemState              `json:"state"`
	CollectionDisposition *CollectionDisposition `json:"collectionDisposition,omitempty"`
	TerminalOutcome       *TerminalOutcome       `json:"terminalOutcome,omitempty"`
	RunID                 *string                `json:"runId,omitempty"`
	RunDeleted            bool                   `json:"runDeleted"`
	RunProvenance         *RunProvenance         `json:"runProvenance,omitempty"`
	Task                  ExactArtifact          `json:"task"`
	ItemOrigin            ItemOrigin             `json:"itemOrigin"`
	Result                *ExactArtifact         `json:"result,omitempty"`
	CreatedAt             time.Time              `json:"createdAt"`
	CollectedAt           *time.Time             `json:"collectedAt,omitempty"`
}

type ProvenanceListParams struct {
	OwnerID   string
	AuditID   string
	FindingID string
	// Optional caller pins bind an HTTP revision envelope or earlier read to
	// the complete page. Unpinned callers still get a consistent page.
	AuditRevision   *uint64
	FindingRevision *uint64
	AfterCreatedAt  *time.Time
	AfterRecordID   string
	Limit           int
}

// FindingRevision names one finding at the exact revision a caller read.
type FindingRevision struct {
	FindingID string
	Revision  uint64
}

// WorkspaceSummary is one database statement's owner-scoped snapshot. Work is
// scoped to the current round; findings and reviews span the entire Audit.
// CompletedChecks, Gaps and Unchecked partition TotalChecks by the coverage
// buckets of auditdomain, the classification the Audit report concludes
// with; Issues is the part of CompletedChecks with an issue found.
// PendingReviews counts only requests that can still be decided.
type WorkspaceSummary struct {
	AuditID            string    `json:"auditId"`
	AuditRevision      uint64    `json:"auditRevision"`
	AsOf               time.Time `json:"asOf"`
	RoundID            *string   `json:"roundId,omitempty"`
	ExecutionState     string    `json:"executionState"`
	OutstandingRuns    int       `json:"outstandingRuns"`
	TotalChecks        int       `json:"totalChecks"`
	CompletedChecks    int       `json:"completedChecks"`
	Issues             int       `json:"issues"`
	Gaps               int       `json:"gaps"`
	Unchecked          int       `json:"unchecked"`
	Findings           int       `json:"findings"`
	UnreviewedFindings int       `json:"unreviewedFindings"`
	PendingReviews     int       `json:"pendingReviews"`
}

type EventListParams struct {
	OwnerID, AuditID string
	// A continuation freezes the immutable prefix; new events appear on refresh.
	ThroughSequence *uint64
	BeforeSequence  *uint64
	Limit           int
}
