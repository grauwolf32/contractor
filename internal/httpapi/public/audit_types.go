package public

import (
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type createAuditRequest struct {
	Labels        runMetadataLabels                `json:"labels,omitempty"`
	Profile       auditProfileSelectorRequest      `json:"profile"`
	Inputs        map[string]contracts.ArtifactRef `json:"inputs"`
	RuntimeLabels []string                         `json:"runtimeLabels,omitempty"`
	Scope         auditservice.Scope               `json:"scope"`
}

type auditProfileSelectorRequest struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type auditProfileResponse struct {
	Ref     auditprofile.AuditProfileRef          `json:"ref"`
	Inputs  map[string]auditcontract.ProgramInput `json:"inputs"`
	Limits  auditcontract.ProgramLimits           `json:"limits"`
	Policy  auditcontract.OperationPolicy         `json:"policy"`
	Program *auditcontract.Program                `json:"program,omitempty"`
}

type auditProfilePageResponse struct {
	Items []auditProfileResponse `json:"items"`
	Page  pageInfoResponse       `json:"page"`
}

type auditRuntimeSnapshotResponse struct {
	Default runtimeconfig.PinnedLabel   `json:"default"`
	Labels  []runtimeconfig.PinnedLabel `json:"labels"`
}

type auditBaselineResponse struct {
	Labels                  runlabels.RunMetadataLabels          `json:"labels"`
	RequiredPlacementLabels []string                             `json:"requiredPlacementLabels"`
	ActiveChecks            contracts.ActiveChecksPolicy         `json:"activeChecks"`
	Inputs                  map[string]auditstore.ExactArtifact  `json:"inputs"`
	Scope                   auditservice.Scope                   `json:"scope"`
	RuntimeLabels           []string                             `json:"runtimeLabels"`
	RuntimeConfig           auditRuntimeSnapshotResponse         `json:"runtimeConfig"`
	Skills                  []auditSkillResponse                 `json:"skills"`
	Standards               []auditstandards.PinnedPackage       `json:"standards"`
	ProjectHTTPTarget       *runtimesettings.HTTPOriginTargetRef `json:"projectHttpTarget,omitempty"`
	Inventory               *auditBaselineInventoryResponse      `json:"inventory,omitempty"`
}

type auditSkillResponse struct {
	Name         string                `json:"name"`
	Source       contracts.ArtifactRef `json:"source"`
	SourceDigest string                `json:"sourceDigest"`
	SourceSize   int64                 `json:"sourceSize"`
}

type auditBaselineInventoryResponse struct {
	SourceContentDigest      string                               `json:"sourceContentDigest"`
	CanonicalInventoryDigest string                               `json:"canonicalInventoryDigest"`
	StandardSelection        *auditprofile.AuditStandardSelection `json:"standardSelection,omitempty"`
	Gaps                     []string                             `json:"gaps"`
	Worklist                 auditstore.ExactArtifact             `json:"worklist"`
}

type auditProfileIdentityResponse struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}

type auditStopReasonResponse struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type auditResponse struct {
	TaskCount               int                                 `json:"taskCount"`
	Iteration               int                                 `json:"iteration"`
	Labels                  runlabels.RunMetadataLabels         `json:"labels"`
	RequiredPlacementLabels []string                            `json:"requiredPlacementLabels"`
	AuditID                 string                              `json:"auditId"`
	ProjectID               string                              `json:"projectId"`
	Profile                 auditProfileIdentityResponse        `json:"profile"`
	Inputs                  map[string]auditstore.ExactArtifact `json:"inputs"`
	Scope                   auditservice.Scope                  `json:"scope"`
	RuntimeLabels           []string                            `json:"runtimeLabels"`
	Baseline                *auditBaselineResponse              `json:"baseline,omitempty"`
	State                   auditstore.AuditState               `json:"state"`
	Revision                uint64                              `json:"revision"`
	DispatchState           auditstore.DispatchState            `json:"dispatchState"`
	HoldState               auditstore.HoldState                `json:"holdState"`
	DeadlineAt              *time.Time                          `json:"deadlineAt,omitempty"`
	PausedAt                *time.Time                          `json:"pausedAt,omitempty"`
	Limits                  auditstore.Limits                   `json:"limits"`
	ReservedRunCount        int                                 `json:"reservedRunCount"`
	SubmittedRunCount       int                                 `json:"submittedRunCount"`
	OutstandingRunCount     int                                 `json:"outstandingRunCount"`
	CollectionPending       bool                                `json:"collectionPending"`
	RetainedEvidenceBytes   int64                               `json:"retainedEvidenceBytes"`
	EventSequence           uint64                              `json:"eventSequence"`
	StopReason              *auditStopReasonResponse            `json:"stopReason,omitempty"`
	CreatedAt               time.Time                           `json:"createdAt"`
	UpdatedAt               time.Time                           `json:"updatedAt"`
	StartedAt               *time.Time                          `json:"startedAt,omitempty"`
	FinishedAt              *time.Time                          `json:"finishedAt,omitempty"`
	DeletionRequestedAt     *time.Time                          `json:"deletionRequestedAt,omitempty"`
}

type auditPageResponse struct {
	Items []auditResponse  `json:"items"`
	Page  pageInfoResponse `json:"page"`
}

type auditReportResponse struct {
	Status          auditservice.ReportStatus `json:"status"`
	MachineArtifact *auditstore.ExactArtifact `json:"machineArtifact,omitempty"`
	SummaryArtifact *auditstore.ExactArtifact `json:"summaryArtifact,omitempty"`
	Machine         json.RawMessage           `json:"machine,omitempty"`
	Summary         string                    `json:"summary,omitempty"`
}

type auditStartResponse struct {
	Audit auditResponse `json:"audit"`
}
