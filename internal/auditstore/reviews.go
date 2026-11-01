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
	ReportAcceptanceReviewKind = "report-acceptance"
)

type ReviewSubjectKind string

const (
	ReviewSubjectItemAction ReviewSubjectKind = "audit-item-action"
	ReviewSubjectReport     ReviewSubjectKind = "audit-report"
)

func (kind ReviewSubjectKind) Valid() bool {
	return kind == ReviewSubjectItemAction ||
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

func (action ReviewRequestedAction) Valid() bool { return ReviewAction(action).Valid() }

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

type ReviewDecision struct {
	DecisionID      string       `json:"decisionId"`
	RequestID       string       `json:"requestId"`
	AuditID         string       `json:"auditId"`
	Action          ReviewAction `json:"action"`
	ActorID         string       `json:"actorId"`
	Rationale       string       `json:"rationale"`
	SubjectRevision uint64       `json:"subjectRevision"`
	SubjectDigest   string       `json:"subjectDigest"`
	CreatedAt       time.Time    `json:"createdAt"`
}

type ReviewRequest struct {
	RequestID        string                  `json:"requestId"`
	AuditID          string                  `json:"auditId"`
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
	RequestDigest string
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
	ProjectRevision *uint64
	AuditRevision   *uint64
	OwnerID         string
	AuditID         string
	State           *FindingState
	Verdict         *AnalystVerdict
	Unreviewed      bool
	Severity        *FindingSeverity
	AfterCreatedAt  *time.Time
	AfterFindingID  string
	Limit           int
}

type ReviewListParams struct {
	AuditRevision  *uint64
	OwnerID        string
	AuditID        string
	State          *ReviewState
	AfterCreatedAt *time.Time
	AfterRequestID string
	Limit          int
}

// Owner lists are live, newest-first keyset pages across active Projects.
// AuditID and AuditRevision are intentionally absent from these contracts.
type OwnerFindingListParams struct {
	ProjectID       string
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
	ProjectRevision uint64    `json:"projectRevision,omitempty"`
	AuditRevision   uint64    `json:"auditRevision"`
	AsOf            time.Time `json:"asOf"`
	Total           int       `json:"total"`
}

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
