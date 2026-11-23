package auditstore

import (
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
)

// ProjectFinding is the single current problem record. Sources and check
// outputs are historical facts and cannot supply its optional analyst fields.
type ProjectFinding struct {
	CorrelationAlgorithm string `json:"-"`
	CorrelationValue     string `json:"-"`
	CorrelationGroupKey  string `json:"-"`
	CorrelationReason    string `json:"-"`

	FindingID         string                               `json:"findingId" strictjson:"required"`
	ProjectID         string                               `json:"projectId" strictjson:"required"`
	FirstReceiptID    string                               `json:"firstReceiptId" strictjson:"required"`
	FirstProposal     ExactArtifact                        `json:"firstProposal" strictjson:"required"`
	Title             string                               `json:"title" strictjson:"required"`
	Claim             *auditdomain.FindingReportProjection `json:"claim,omitempty"`
	State             *FindingState                        `json:"state" strictjson:"required,nullable"`
	AnalystVerdict    *AnalystVerdict                      `json:"analystVerdict" strictjson:"required,nullable"`
	AnalystSeverity   *FindingSeverity                     `json:"analystSeverity" strictjson:"required,nullable"`
	Rationale         *string                              `json:"rationale" strictjson:"required,nullable"`
	ActorID           *string                              `json:"actorId" strictjson:"required,nullable"`
	AssessedAt        *time.Time                           `json:"assessedAt" strictjson:"required,nullable"`
	DuplicateTargetID *string                              `json:"duplicateTargetId" strictjson:"required,nullable"`
	Revision          uint64                               `json:"revision" strictjson:"required"`
	CreatedAt         time.Time                            `json:"createdAt" strictjson:"required"`
	UpdatedAt         time.Time                            `json:"updatedAt" strictjson:"required"`
}

// ProjectFindingEdit is an ordinary owner command, independent of Audit reviews.
type ProjectFindingEdit struct {
	OwnerID           string
	ProjectID         string
	FindingID         string
	ExpectedRevision  uint64
	Verdict           AnalystVerdict
	Severity          *FindingSeverity
	Rationale         string
	DuplicateTargetID *string
}

// ProjectFindingSummary is a compact canonical-state page row. Full claims,
// rationale and independent source histories have separate detail reads.
type ProjectFindingSummary struct {
	FindingID          string           `json:"findingId" strictjson:"required"`
	ProjectID          string           `json:"projectId" strictjson:"required"`
	Title              string           `json:"title" strictjson:"required"`
	TitleTruncated     bool             `json:"titleTruncated" strictjson:"required"`
	SubjectKind        string           `json:"subjectKind" strictjson:"required"`
	SubjectKey         string           `json:"subjectKey" strictjson:"required"`
	SubjectTruncated   bool             `json:"subjectTruncated" strictjson:"required"`
	SeveritySuggestion string           `json:"severitySuggestion" strictjson:"required"`
	State              *FindingState    `json:"state" strictjson:"required,nullable"`
	AnalystVerdict     *AnalystVerdict  `json:"analystVerdict" strictjson:"required,nullable"`
	AnalystSeverity    *FindingSeverity `json:"analystSeverity" strictjson:"required,nullable"`
	DuplicateTargetID  *string          `json:"duplicateTargetId" strictjson:"required,nullable"`
	Revision           uint64           `json:"revision" strictjson:"required"`
	CreatedAt          time.Time        `json:"createdAt" strictjson:"required"`
	UpdatedAt          time.Time        `json:"updatedAt" strictjson:"required"`
}

// FindingPageBasis fences a page without rescanning its whole collection.
// Exact aggregate counts are read separately by the workspace summary.
type FindingPageBasis struct {
	AuditRevision   uint64    `json:"auditRevision" strictjson:"required"`
	ProjectRevision uint64    `json:"projectRevision" strictjson:"required"`
	AsOf            time.Time `json:"asOf" strictjson:"required"`
}

// FindingWindow bounds examined canonical rows independently of match density.
// Its cursor may advance even when no visible row matched the current filters.
type FindingWindow struct {
	Items          []ProjectFindingSummary
	ContinueBefore *ProjectFindingSummary
}

// PresentationSeverity uses the analyst rating when present, otherwise the producer suggestion.
func (row ProjectFindingSummary) PresentationSeverity() FindingSeverity {
	if row.AnalystSeverity != nil {
		return *row.AnalystSeverity
	}
	return FindingSeverity(row.SeveritySuggestion)
}

// NeedsReview is the shared unsettled-record predicate, excluding informational material.
func (row ProjectFindingSummary) NeedsReview() bool {
	unsettled := row.AnalystVerdict == nil || row.State != nil && *row.State == FindingProposed
	return unsettled && row.PresentationSeverity() != SeverityInformational
}

func (row ProjectFindingSummary) Matches(state *FindingState, verdict *AnalystVerdict, needsReview bool, severity *FindingSeverity) bool {
	return (state == nil || row.State != nil && *row.State == *state) &&
		(verdict == nil || row.AnalystVerdict != nil && *row.AnalystVerdict == *verdict) &&
		(!needsReview || row.NeedsReview()) &&
		(severity == nil || row.PresentationSeverity() == *severity)
}
