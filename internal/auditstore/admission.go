package auditstore

import (
	"context"
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
)

const (
	AdmissionSchema            = "contractor.audit.admission.v2"
	AdmissionProvenanceSchema  = "contractor.audit.admission-provenance.v1"
	AdmissionDecisionSchema    = "contractor.audit.admission-decision.v1"
	MaxAdmissionPrimary        = auditdomain.MaximumPreparedRoundPageItems
	MaxAdmissionRelated        = auditdomain.MaximumPreparedRoundRelated
	MaxAdmissionMetadataBytes  = 1 << 20
	MaxAdmissionPayloadBytes   = 64 << 20
	MaxAdmissionProgressBytes  = 64 << 10
	AdmissionValidationTimeout = 10 * time.Minute
)

type AdmissionCursor struct {
	CreatedAt time.Time `json:"created_at" strictjson:"required"`
	ReceiptID string    `json:"receipt_id" strictjson:"required"`
	Ordinal   int       `json:"ordinal" strictjson:"required"`
}

type AdmissionProgress struct {
	Schema          string          `json:"schema" strictjson:"required"`
	Generation      string          `json:"generation" strictjson:"required"`
	Revision        uint64          `json:"revision" strictjson:"required"`
	AuditID         string          `json:"audit_id" strictjson:"required"`
	PreviousRoundID string          `json:"previous_round_id" strictjson:"required"`
	RoundID         string          `json:"round_id" strictjson:"required"`
	RoundOrdinal    int             `json:"round_ordinal" strictjson:"required"`
	ProfileDigest   string          `json:"profile_digest" strictjson:"required"`
	BaselineDigest  string          `json:"baseline_digest" strictjson:"required"`
	SourceCutoffAt  time.Time       `json:"source_cutoff_at" strictjson:"required"`
	SourceUpper     AdmissionCursor `json:"source_upper" strictjson:"required"`
	Cursor          AdmissionCursor `json:"cursor" strictjson:"required"`
	Budget          json.RawMessage `json:"budget" strictjson:"required"`
	SelectedCount   int             `json:"selected_count" strictjson:"required"`
	PublishedPages  int             `json:"published_pages" strictjson:"required"`
	StagedPages     int             `json:"staged_pages" strictjson:"required"`
	PagesDigest     string          `json:"pages_digest" strictjson:"required"`
	StagedCount     int             `json:"staged_count" strictjson:"required"`
	Exhausted       bool            `json:"exhausted" strictjson:"required"`
	StopCode        string          `json:"stop_code,omitempty"`
	Root            *ExactArtifact  `json:"root" strictjson:"required,nullable"`
	DeadlineAt      time.Time       `json:"deadline_at" strictjson:"required"`
	Changed         bool            `json:"-"`
}

type AdmissionStepParams struct {
	Claim           ControllerClaim
	PreviousRoundID string
	Now             time.Time
}

// AdmissionDecision is server-owned exact source attribution. C publishes its
// canonical JSON and an indexed ArtifactLink under a deterministic natural key.
// Permanent rejections omit the generation from that key; deferrals do not.
type AdmissionDecision struct {
	ReceiptID            string        `json:"receipt_id" strictjson:"required"`
	ProposedCheckOrdinal int           `json:"proposed_check_ordinal" strictjson:"required"`
	Proposal             ExactArtifact `json:"proposal" strictjson:"required"`
	SourceCheckDigest    string        `json:"source_check_digest" strictjson:"required"`
	PlanDigest           string        `json:"plan_digest,omitempty"`
	Disposition          string        `json:"disposition" strictjson:"required"`
	Code                 string        `json:"code" strictjson:"required"`
	WorkflowRole         string        `json:"workflow_role,omitempty"`
	RouteDigest          string        `json:"route_digest,omitempty"`
	ItemID               string        `json:"item_id,omitempty"`
}

// AdmissionMutation is one changing phase. A callback writes immutable source
// artifacts through its transaction-bound service, then returns their exact
// descriptors here. The store commits links, counters and checkpoint together.
type AdmissionMutation struct {
	Next             AdmissionProgress
	Page             *ExactArtifact
	PageContent      json.RawMessage
	Root             *ExactArtifact
	StagePageOrdinal *int
	Retained         []ArtifactLink
	Decisions        []AdmissionDecision
	Restart          bool
}

type AdmissionCandidate struct {
	ReceiptID       string
	Proposal        ExactArtifact
	CreatedAt       time.Time
	Ordinal         int
	Header          auditdomain.FindingCheckProjection
	ProjectionBytes int
	AlreadyConsumed bool
	AlreadyRejected bool
	LateSource      bool
	NoChecks        bool
	Cursor          AdmissionCursor
}

type AdmissionCandidatePage struct {
	Candidates  []AdmissionCandidate
	Exhausted   bool
	LateSources bool
}

type AdmissionFindingSource struct {
	ReceiptID             string
	ProjectID             string
	Original              ExactArtifact
	Retained              ExactArtifact
	OriginalEvidence      []ExactArtifact
	RetainedEvidence      []ExactArtifact
	OriginRunID           string
	OriginExecutionID     string
	OriginAuditID         string
	EvidenceLimitExceeded bool
	Projection            auditdomain.FindingChecksProjection
}

type AdmissionFindingCheck struct {
	Source  AdmissionFindingSource
	Check   auditdomain.ProposedCheck
	Plan    auditdomain.CheckPlan
	Context auditdomain.FindingCheckContext
}

// AdmissionReader is deliberately narrower than the query/controller seams.
// Every implementation is tied to the same claimed admission transaction.
type AdmissionReader interface {
	ReadCandidates(context.Context, AdmissionCursor, int) (AdmissionCandidatePage, error)
	FindingSource(context.Context, string) (AdmissionFindingSource, error)
	ReadFindingCheck(context.Context, string, int) (AdmissionFindingCheck, error)
	ReadPage(context.Context, int) (ExactArtifact, []byte, error)
	GetPreparationOutput(context.Context, string, string, string) (AcceptedPreparationOutput, error)
	GetArtifactLink(context.Context, string, string) (ArtifactLink, error)
	DependencyResultSource(context.Context, string, string, string) (DependencyResultSource, error)
}
