package auditstore

import (
	"strings"
	"time"
	"unicode/utf8"
)

// ValidFindingIdentity validates identities shared by canonical Findings,
// their owner/Project/Audit scope and source/history reads: non-blank UTF-8
// without NUL, at most maximum bytes.
func ValidFindingIdentity(value string, maximum int) bool {
	return strings.TrimSpace(value) != "" && utf8.ValidString(value) &&
		!strings.ContainsRune(value, 0) && len([]byte(value)) <= maximum
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

// Owner lists are live, newest-first keyset pages across active Projects.
// AuditID and AuditRevision are intentionally absent from these contracts.
type OwnerFindingListParams struct {
	ProjectID       string
	OwnerID         string
	State           *FindingState
	Verdict         *AnalystVerdict
	Unreviewed      bool
	Severity        *FindingSeverity
	BeforeCreatedAt *time.Time
	BeforeFindingID string
	Limit           int
}

type FindingRevision struct {
	FindingID string
	Revision  uint64
}

// WorkspaceSummary is one database statement's owner-scoped snapshot. Work is
// scoped to the current round; findings span the entire Audit.
// CompletedChecks, Gaps and Unchecked partition TotalChecks by the coverage
// buckets of auditdomain, the classification the Audit report concludes
// with; Issues is the part of CompletedChecks with an issue found.
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
}

type EventListParams struct {
	OwnerID, AuditID string
	// A continuation freezes the immutable prefix; new events appear on refresh.
	ThroughSequence *uint64
	BeforeSequence  *uint64
	Limit           int
}
