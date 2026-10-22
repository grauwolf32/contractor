package auditstore

import (
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
)

const (
	FindingComparisonDefaultLimit  = 20
	FindingComparisonMaximumLimit  = 100
	FindingComparisonMaximumBytes  = 256 << 10
	FindingComparisonWorkRows      = 1000
	FindingCandidateWindowRows     = FindingComparisonWorkRows - FindingComparisonMaximumLimit - 3
	FindingCandidateMaximumReasons = 8
)

type FindingComparisonProfile struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}

type FindingComparisonAudit struct {
	AuditID                string                   `json:"auditId"`
	Revision               uint64                   `json:"revision"`
	State                  AuditState               `json:"state"`
	Profile                FindingComparisonProfile `json:"profile"`
	BaselineDigest         *string                  `json:"baselineDigest,omitempty"`
	BaselineDigestEncoding string                   `json:"baselineDigestEncoding"`
	Coverage               string                   `json:"coverage"`
}

type FindingCorrelationFence struct {
	ProjectID       string                   `json:"projectId"`
	ProjectRevision uint64                   `json:"projectRevision,omitempty"`
	Audits          []FindingComparisonAudit `json:"audits"`
}

// FindingMatchHints contains only explicitly cached, bounded deterministic
// inputs. Text descriptions/titles are never matching inputs.
type FindingMatchHints struct {
	Subject           *auditdomain.FindingSubject     `json:"subject,omitempty"`
	Families          []auditdomain.StandardReference `json:"families" strictjson:"required"`
	FamiliesAvailable bool                            `json:"familiesAvailable" strictjson:"required"`
	ProducerKey       string                          `json:"producerKey" strictjson:"required"`
	Sites             []string                        `json:"sites" strictjson:"required"`
	Repository        string                          `json:"repository" strictjson:"required"`
	Commit            string                          `json:"commit" strictjson:"required"`
}

type FindingCandidateSource struct {
	AuditRevision     uint64
	ProjectRevision   uint64
	FindingID         string
	AuditID           string
	ProjectID         string
	FirstReceiptID    string
	SelectedReceiptID string
	GroupKey          string
	Hints             *FindingMatchHints
	HintsReason       string
}

type FindingCandidateSeek struct {
	GroupKey         string
	OwnerID          string
	ProjectID        string
	AuditID          string
	ExcludeFindingID string
	AfterCreatedAt   *time.Time
	AfterFindingID   string
}

type FindingCandidateRecord struct {
	FindingID      string
	AuditID        string
	FirstReceiptID string
	GroupKey       string
	CreatedAt      time.Time
	Hints          *FindingMatchHints
	HintsReason    string
}

type FindingComparisonOrigin struct {
	ReceiptID             string `json:"receiptId"`
	RunID                 string `json:"runId"`
	RunDeleted            bool   `json:"runDeleted"`
	LogicalAgentName      string `json:"logicalAgentName"`
	WorkflowName          string `json:"workflowName"`
	WorkflowVersion       string `json:"workflowVersion"`
	WorkflowClosureDigest string `json:"workflowClosureDigest"`
}

// FindingComparisonPreview is deliberately a display preview, not a proposal
// or evidence document. Finding detail retains the full original material.
type FindingComparisonPreview struct {
	CorrelationRole    string                  `json:"correlationRole,omitempty"`
	FindingID          string                  `json:"findingId"`
	AuditID            string                  `json:"auditId"`
	Revision           uint64                  `json:"revision"`
	Preview            bool                    `json:"preview"`
	Title              string                  `json:"title"`
	TitleTruncated     bool                    `json:"titleTruncated"`
	State              FindingState            `json:"state"`
	SeveritySuggestion string                  `json:"severitySuggestion"`
	AnalystVerdict     *AnalystVerdict         `json:"analystVerdict,omitempty"`
	AnalystSeverity    *FindingSeverity        `json:"analystSeverity,omitempty"`
	AssessmentID       *string                 `json:"assessmentId,omitempty"`
	DecisionID         *string                 `json:"decisionId,omitempty"`
	SemanticAssessment *string                 `json:"semanticAssessment,omitempty"`
	Origin             FindingComparisonOrigin `json:"origin"`
}

type FindingComparisonGroup struct {
	GroupKey          string
	Algorithm         string
	Value             string
	LeftFindingID     *string
	RightFindingID    *string
	UnavailableReason string
}

type FindingComparisonSeek struct {
	OwnerID      string
	ProjectID    string
	LeftAuditID  string
	RightAuditID string
	Unmatched    bool
	AfterKey     string
	Limit        int
}

type FindingMemberSeek struct {
	OwnerID        string
	ProjectID      string
	GroupKey       string
	AuditIDs       []string
	AfterFindingID string
	Limit          int
}

type FindingComparisonProjection struct {
	Schema       string `json:"schema" strictjson:"required"`
	Availability string `json:"availability" strictjson:"required"`
	Reason       string `json:"reason,omitempty"`
	Preview      *struct {
		Title              string `json:"title" strictjson:"required"`
		TitleTruncated     bool   `json:"titleTruncated" strictjson:"required"`
		SeveritySuggestion string `json:"severitySuggestion" strictjson:"required"`
	} `json:"preview,omitempty"`
	Hints *FindingMatchHints `json:"hints,omitempty"`
}
