package public

import (
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type createAuditRequest struct {
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
	Ref                     config.AuditProfileRef                  `json:"ref"`
	Mode                    config.AuditProfileMode                 `json:"mode"`
	Standards               []config.AuditStandardRef               `json:"standards"`
	Inputs                  map[string]config.AuditProfileInput     `json:"inputs"`
	Inventory               config.AuditInventory                   `json:"inventory"`
	Workflows               map[string]auditProfileWorkflowResponse `json:"workflows,omitempty"`
	Execution               config.AuditExecutionPolicy             `json:"execution"`
	Interaction             config.AuditInteractionPolicy           `json:"interaction"`
	ServerCompatible        bool                                    `json:"serverCompatible"`
	RequiresInputValidation bool                                    `json:"requiresInputValidation"`
	CompatibilityReasons    []auditservice.CompatibilityReason      `json:"compatibilityReasons"`
}

type auditProfileWorkflowResponse struct {
	MaxRunAttempts int                                             `json:"maxRunAttempts,omitempty"`
	Kind           config.AuditWorkflowRoleKind                    `json:"kind"`
	Workflow       config.WorkflowRef                              `json:"workflow"`
	Inputs         map[string]config.AuditWorkflowInputMapping     `json:"inputs"`
	Parameters     map[string]config.AuditWorkflowParameterMapping `json:"parameters"`
	Outputs        map[string]string                               `json:"outputs"`
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
	Inputs            map[string]auditstore.ExactArtifact `json:"inputs"`
	Scope             auditservice.Scope                  `json:"scope"`
	RuntimeLabels     []string                            `json:"runtimeLabels"`
	RuntimeConfig     auditRuntimeSnapshotResponse        `json:"runtimeConfig"`
	Skills            []auditSkillResponse                `json:"skills"`
	Standards         []auditstandards.PinnedPackage      `json:"standards"`
	ProjectHTTPTarget *contracts.HTTPOriginTargetRef      `json:"projectHttpTarget,omitempty"`
	Inventory         *auditBaselineInventoryResponse     `json:"inventory,omitempty"`
}

type auditSkillResponse struct {
	Name         string                `json:"name"`
	Source       contracts.ArtifactRef `json:"source"`
	SourceDigest string                `json:"sourceDigest"`
	SourceSize   int64                 `json:"sourceSize"`
}

type auditBaselineInventoryResponse struct {
	SourceContentDigest      string                         `json:"sourceContentDigest"`
	CanonicalInventoryDigest string                         `json:"canonicalInventoryDigest"`
	StandardSelection        *config.AuditStandardSelection `json:"standardSelection,omitempty"`
	Gaps                     []string                       `json:"gaps"`
	Worklist                 auditstore.ExactArtifact       `json:"worklist"`
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
	Phase                 auditdomain.AuditPhase              `json:"phase"`
	Preparation           *auditPreparationResponse           `json:"preparation,omitempty"`
	AuditID               string                              `json:"auditId"`
	ProjectID             string                              `json:"projectId"`
	Profile               auditProfileIdentityResponse        `json:"profile"`
	Inputs                map[string]auditstore.ExactArtifact `json:"inputs"`
	Scope                 auditservice.Scope                  `json:"scope"`
	RuntimeLabels         []string                            `json:"runtimeLabels"`
	Baseline              *auditBaselineResponse              `json:"baseline,omitempty"`
	State                 auditstore.AuditState               `json:"state"`
	Revision              uint64                              `json:"revision"`
	CurrentRoundID        *string                             `json:"currentRoundId,omitempty"`
	DispatchState         auditstore.DispatchState            `json:"dispatchState"`
	HoldState             auditstore.HoldState                `json:"holdState"`
	DeadlineAt            *time.Time                          `json:"deadlineAt,omitempty"`
	PausedAt              *time.Time                          `json:"pausedAt,omitempty"`
	Limits                auditstore.Limits                   `json:"limits"`
	ReservedRunCount      int                                 `json:"reservedRunCount"`
	SubmittedRunCount     int                                 `json:"submittedRunCount"`
	OutstandingRunCount   int                                 `json:"outstandingRunCount"`
	RetainedEvidenceBytes int64                               `json:"retainedEvidenceBytes"`
	EventSequence         uint64                              `json:"eventSequence"`
	StopReason            *auditStopReasonResponse            `json:"stopReason,omitempty"`
	CreatedAt             time.Time                           `json:"createdAt"`
	UpdatedAt             time.Time                           `json:"updatedAt"`
	StartedAt             *time.Time                          `json:"startedAt,omitempty"`
	FinishedAt            *time.Time                          `json:"finishedAt,omitempty"`
	DeletionRequestedAt   *time.Time                          `json:"deletionRequestedAt,omitempty"`
}

type auditPageResponse struct {
	Items []auditResponse  `json:"items"`
	Page  pageInfoResponse `json:"page"`
}

type auditRoundResponse struct {
	RoundID           string                   `json:"roundId"`
	Ordinal           int                      `json:"ordinal"`
	Manifest          auditstore.ExactArtifact `json:"manifest"`
	State             auditstore.RoundState    `json:"state"`
	ExpectedItemCount int                      `json:"expectedItemCount"`
	Revision          uint64                   `json:"revision"`
	CreatedAt         time.Time                `json:"createdAt"`
	UpdatedAt         time.Time                `json:"updatedAt"`
}

type auditItemResponse struct {
	ItemID              string                       `json:"itemId"`
	RoundID             string                       `json:"roundId"`
	ItemKey             string                       `json:"itemKey"`
	Ordinal             int                          `json:"ordinal"`
	Kind                string                       `json:"kind"`
	SubjectKey          string                       `json:"subjectKey"`
	Task                auditstore.ExactArtifact     `json:"task"`
	Origin              auditstore.ItemOrigin        `json:"origin"`
	WorkflowRole        string                       `json:"workflowRole"`
	State               auditstore.ItemState         `json:"state"`
	ApprovalKind        auditstore.ItemApprovalKind  `json:"approvalKind"`
	ApprovalDigest      string                       `json:"approvalDigest,omitempty"`
	FinalDisposition    *auditstore.FinalDisposition `json:"finalDisposition,omitempty"`
	AcceptedResult      *auditstore.ExactArtifact    `json:"acceptedResult,omitempty"`
	LastExecutionItemID *string                      `json:"lastExecutionItemId,omitempty"`
	Attempts            []auditstore.ItemAttempt     `json:"attempts"`
	CreatedAt           time.Time                    `json:"createdAt"`
	UpdatedAt           time.Time                    `json:"updatedAt"`
}

type auditItemPageResponse struct {
	Items []auditItemResponse `json:"items"`
	Page  pageInfoResponse    `json:"page"`
}

type auditCoverageResponse struct {
	RoundID    string                      `json:"roundId"`
	ItemID     string                      `json:"itemId"`
	Ordinal    int                         `json:"ordinal"`
	ItemKey    string                      `json:"itemKey"`
	SubjectKey string                      `json:"subjectKey"`
	Coverage   auditstore.Coverage         `json:"coverage"`
	Result     *auditstore.ExactArtifact   `json:"result,omitempty"`
	UpdatedAt  time.Time                   `json:"updatedAt"`
	Details    *auditstore.CoverageDetails `json:"details,omitempty"`
}

type auditCoveragePageResponse struct {
	Items []auditCoverageResponse `json:"items"`
	Page  pageInfoResponse        `json:"page"`
}

type auditReportResponse struct {
	Status          auditservice.ReportStatus   `json:"status"`
	Review          *auditservice.ReviewRequest `json:"review,omitempty"`
	MachineArtifact *auditstore.ExactArtifact   `json:"machineArtifact,omitempty"`
	SummaryArtifact *auditstore.ExactArtifact   `json:"summaryArtifact,omitempty"`
	Machine         json.RawMessage             `json:"machine,omitempty"`
	Summary         string                      `json:"summary,omitempty"`
}

type auditStartResponse struct {
	Audit auditResponse       `json:"audit"`
	Round *auditRoundResponse `json:"round,omitempty"`
	Items []auditItemResponse `json:"items"`
}
