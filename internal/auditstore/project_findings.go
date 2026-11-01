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

	FindingID         string                               `json:"findingId"`
	ProjectID         string                               `json:"projectId"`
	FirstReceiptID    string                               `json:"firstReceiptId"`
	FirstProposal     ExactArtifact                        `json:"firstProposal"`
	Title             string                               `json:"title"`
	Claim             *auditdomain.FindingReportProjection `json:"claim,omitempty"`
	State             *FindingState                        `json:"state"`
	AnalystVerdict    *AnalystVerdict                      `json:"analystVerdict"`
	AnalystSeverity   *FindingSeverity                     `json:"analystSeverity"`
	Rationale         *string                              `json:"rationale"`
	ActorID           *string                              `json:"actorId"`
	AssessedAt        *time.Time                           `json:"assessedAt"`
	DuplicateTargetID *string                              `json:"duplicateTargetId"`
	Revision          uint64                               `json:"revision"`
	CreatedAt         time.Time                            `json:"createdAt"`
	UpdatedAt         time.Time                            `json:"updatedAt"`
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
	FindingID         string           `json:"findingId"`
	ProjectID         string           `json:"projectId"`
	Title             string           `json:"title"`
	TitleTruncated    bool             `json:"titleTruncated"`
	State             *FindingState    `json:"state"`
	AnalystVerdict    *AnalystVerdict  `json:"analystVerdict"`
	AnalystSeverity   *FindingSeverity `json:"analystSeverity"`
	DuplicateTargetID *string          `json:"duplicateTargetId"`
	Revision          uint64           `json:"revision"`
	CreatedAt         time.Time        `json:"createdAt"`
	UpdatedAt         time.Time        `json:"updatedAt"`
}

// FindingPageBasis fences a page without rescanning its whole collection.
// Exact aggregate counts are read separately by the workspace summary.
type FindingPageBasis struct {
	AuditRevision   uint64    `json:"auditRevision"`
	ProjectRevision uint64    `json:"projectRevision"`
	AsOf            time.Time `json:"asOf"`
}
