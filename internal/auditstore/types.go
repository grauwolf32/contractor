package auditstore

import (
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const (
	MaxClaimBatch           = 100
	MaxReconcileRows        = 200
	MaxSnapshotBytes        = auditprofile.MaxAuditSnapshotBytes
	MaxSummaryBytes         = 64 << 10
	MaxCollectionItems      = auditcontract.MaxExecutionMembers
	MaxArtifactLinksPerCall = 1024
	MaxArtifactRefBytes     = 4 << 10
	MaxRetainedRefsBytes    = 8 << 20

	// MaxStopReasonMessageBytes bounds a stop reason message, as the
	// audits_stop_reason_message_check constraint does.
	MaxStopReasonMessageBytes = 4 << 10
	// MaxArtifactLogicalKeyBytes bounds the logical key of an Audit receipt
	// link in UTF-8 bytes (LIM-AUD-39).
	MaxArtifactLogicalKeyBytes = 512
)

type AuditState string

const (
	AuditDraft      AuditState = "draft"
	AuditActive     AuditState = "active"
	AuditPaused     AuditState = "paused"
	AuditFinalizing AuditState = "finalizing"
	AuditCancelling AuditState = "cancelling"
	AuditCompleted  AuditState = "completed"
	AuditCancelled  AuditState = "cancelled"
	AuditFailed     AuditState = "failed"
	AuditDeleting   AuditState = "deleting"
)

func (s AuditState) Valid() bool {
	switch s {
	case AuditDraft, AuditActive, AuditPaused,
		AuditFinalizing, AuditCancelling, AuditCompleted, AuditCancelled,
		AuditFailed, AuditDeleting:
		return true
	default:
		return false
	}
}

func (s AuditState) Terminal() bool {
	return s == AuditCompleted || s == AuditCancelled || s == AuditFailed
}

type DispatchState string

const (
	DispatchOpen   DispatchState = "open"
	DispatchClosed DispatchState = "closed"
)

type HoldState string

const (
	HoldPending  HoldState = "pending"
	HoldHeld     HoldState = "held"
	HoldReleased HoldState = "released"
)

type ExecutionState string

const (
	ExecutionIntent     ExecutionState = "intent"
	ExecutionSubmitted  ExecutionState = "submitted"
	ExecutionCollecting ExecutionState = "collecting"
	ExecutionCollected  ExecutionState = "collected"
)

func (s ExecutionState) Valid() bool {
	return s == ExecutionIntent || s == ExecutionSubmitted ||
		s == ExecutionCollecting || s == ExecutionCollected
}

type TerminalOutcome string

const (
	TerminalSucceeded        TerminalOutcome = "succeeded"
	TerminalFailed           TerminalOutcome = "failed"
	TerminalCancelled        TerminalOutcome = "cancelled"
	TerminalSubmissionFailed TerminalOutcome = "submission-failed"
)

func (o TerminalOutcome) Valid() bool {
	return o == TerminalSucceeded || o == TerminalFailed ||
		o == TerminalCancelled || o == TerminalSubmissionFailed
}

type CollectionDisposition string

const (
	CollectionAccepted           CollectionDisposition = "accepted-result"
	CollectionMissingOutput      CollectionDisposition = "missing-output"
	CollectionInvalidResult      CollectionDisposition = "invalid-result"
	CollectionExecutionFailed    CollectionDisposition = "execution-failed"
	CollectionExecutionCancelled CollectionDisposition = "execution-cancelled"
	CollectionContractInvalid    CollectionDisposition = "collection-contract-invalid"
)

func (d CollectionDisposition) Valid() bool {
	return d == CollectionAccepted || d == CollectionMissingOutput ||
		d == CollectionInvalidResult || d == CollectionExecutionFailed ||
		d == CollectionExecutionCancelled || d == CollectionContractInvalid
}

type Limits struct {
	MaxRounds          int   `json:"maxRounds"`
	BatchSize          int   `json:"batchSize"`
	MaxItemsPerRound   int   `json:"maxItemsPerRound"`
	MaxItemsTotal      int   `json:"maxItemsTotal"`
	MaxSubmittedRuns   int   `json:"maxSubmittedRuns"`
	MaxItemRunAttempts int   `json:"maxItemRunAttempts"`
	MaxEvidenceBytes   int64 `json:"maxEvidenceBytes"`
}

type ProfileIdentity struct {
	Name    string
	Version string
	Digest  string
}

type StopReason struct {
	Code    string
	Message string
}

type Audit struct {
	AuditID          string
	OwnerID          string
	ProjectID        string
	Profile          ProfileIdentity
	ProfileSnapshot  json.RawMessage
	InputSelection   json.RawMessage
	BaselineSnapshot json.RawMessage
	State            AuditState

	Revision uint64

	Dispatch                DispatchState
	Hold                    HoldState
	DeadlineAt              *time.Time
	PausedAt                *time.Time
	Limits                  Limits
	ReservedRunCount        int
	SubmittedRunCount       int
	TaskCount               int
	TaskIteration           int
	IterationTaskCount      int
	InitialHypothesisOffset int
	InitialWorkOffset       int
	// OutstandingRunCount counts every child Run that holds a dispatch
	// reservation; OutstandingToolRunCount is its `tool` class part
	// (AUX-R3).
	OutstandingRunCount     int
	OutstandingToolRunCount int
	// CollectionPending is derived in the read snapshot from terminal
	// executions whose output collection has not committed yet.
	CollectionPending     bool
	RetainedEvidenceBytes int64
	EventSequence         uint64
	StopReason            *StopReason
	CreatedAt             time.Time
	UpdatedAt             time.Time
	StartedAt             *time.Time
	FinishedAt            *time.Time
	DeletionRequestedAt   *time.Time
}

type CreateDraftParams struct {
	AuditID         string
	OwnerID         string
	ProjectID       string
	Profile         ProfileIdentity
	ProfileSnapshot json.RawMessage
	InputSelection  json.RawMessage
	Limits          Limits
	IdempotencyKey  string
	RequestDigest   string
}

type ListParams struct {
	OwnerID   string
	ProjectID *string
	// States filters by Audit state; nil lists every state.
	States          []AuditState
	VisibleProjects bool
	ProfileName     *string
	ProfileVersion  *string
	BeforeCreatedAt *time.Time
	BeforeAuditID   string
	Limit           int
}

type MutationOperation string

const (
	MutationCreate          MutationOperation = "audit.create"
	MutationStart           MutationOperation = "audit.start"
	MutationTransition      MutationOperation = "audit.transition"
	MutationDelete          MutationOperation = "audit.delete"
	MutationAcceptTasks     MutationOperation = "audit.tasks.accept"
	MutationWriteHypotheses MutationOperation = "audit.hypotheses.write"
)

type TransitionParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	ExpectedState    AuditState
	TargetState      AuditState
	Reason           *StopReason
	IdempotencyKey   string
	RequestDigest    string
}

// ResumeParams resumes a paused Audit under a fresh dispatch hold. A nil
// DeadlineAt resumes without a time limit.
type ResumeParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	DeadlineAt       *time.Time
	IdempotencyKey   string
	RequestDigest    string
}

type DeleteParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	IdempotencyKey   string
	RequestDigest    string
}

type ClaimedTransitionParams struct {
	Claim            ControllerClaim
	ExpectedRevision uint64
	ExpectedState    AuditState
	TargetState      AuditState
	Reason           *StopReason
}

// ExactArtifact is the shared exact-content descriptor. Tasks, executions
// and results persist only its ref and digest and read back digest-only.
type ExactArtifact = contracts.ExactArtifact

type ControllerClaim struct {
	AuditID   string
	HolderID  string
	Epoch     uint64
	ClaimedAt time.Time
	ExpiresAt time.Time
}

type ClaimParams struct {
	HolderID string
	Lease    time.Duration
	Limit    int
}

type Execution struct {
	ExecutionID string
	AuditID     string

	WorkflowRole string

	SubmissionKey         string
	RequestDigest         string
	RunID                 *string
	State                 ExecutionState
	TerminalOutcome       *TerminalOutcome
	TerminalRunGeneration *string
	TerminalRunSequence   *uint64
	TerminalObservedAt    *time.Time
	RunProvenance         *RunProvenance
	RunDeletedAt          *time.Time
	CreatedAt             time.Time
	UpdatedAt             time.Time
}

// RunProvenance is a bounded, non-secret tombstone captured when a child Run
// is bound. It deliberately stores configuration identity and a closure digest
// rather than the resolved Workflow body, credential material, or live grants.
type RunProvenance struct {
	Schema   string              `json:"schema"`
	RunID    string              `json:"runId"`
	Workflow *WorkflowProvenance `json:"workflow,omitempty"`
}

type WorkflowProvenance struct {
	Name             string `json:"name"`
	Version          string `json:"version"`
	SchemaVersion    string `json:"schemaVersion"`
	ConfigurationRef struct {
		Name    string `json:"name"`
		Version string `json:"version"`
	} `json:"configurationRef"`
	ClosureDigest string `json:"closureDigest"`
}

type BindRunParams struct {
	Claim       ControllerClaim
	ExecutionID string
	RunID       string
}

// RunCreationIntent is the claim-bound immutable authority consumed by the
// trusted Run Service. It contains no secret material and is never exposed by
// the owner-facing repository or public API.
type RunCreationIntent struct {
	Observations            []ContinuationSource
	CanDispatch             bool
	Tasks                   []Task
	ProgramSnapshot         json.RawMessage
	BaselineSnapshot        json.RawMessage
	AuditState              AuditState
	DispatchState           DispatchState
	DeadlineAt              *time.Time
	Labels                  map[string]string
	RequiredPlacementLabels []string
	WorkflowBinding         json.RawMessage
	OwnerID                 string
	ProjectID               string
	Execution               Execution
}

type ObserveTerminalParams struct {
	Claim       ControllerClaim
	ExecutionID string
	RunID       string
	Generation  string
	Sequence    uint64
}

type ObserveSubmissionFailureParams struct {
	Claim       ControllerClaim
	ExecutionID string
}

type ArtifactLink struct {
	Scope            artifacts.ScopeKind `json:"scope"`
	LogicalKey       string              `json:"logicalKey"`
	Artifact         ExactArtifact       `json:"artifact"`
	SourceProvenance json.RawMessage     `json:"sourceProvenance"`
	DisplayRef       string              `json:"displayRef,omitempty"`
	CreatedAt        time.Time           `json:"createdAt,omitempty"`
}

type CollectParams struct {
	Tasks         []TaskCollection
	Claim         ControllerClaim
	ReceiptID     string
	ExecutionID   string
	Disposition   CollectionDisposition
	SourceOutput  *ExactArtifact
	Retained      []ArtifactLink
	ErrorCode     *string
	RequestDigest string
}

type CollectionReceipt struct {
	ReceiptID             string
	AuditID               string
	ExecutionID           string
	RunID                 *string
	TerminalOutcome       TerminalOutcome
	TerminalRunGeneration *string
	TerminalRunSequence   *uint64
	Disposition           CollectionDisposition
	SourceOutput          *ExactArtifact
	Retained              []ArtifactLink
	ErrorCode             *string
	RequestDigest         string
	CreatedAt             time.Time
}

// CollectionReceiptSummary deliberately omits retained/source JSON so one
// bounded reconciliation scan cannot load hundreds of multi-megabyte receipt
// payloads. Exact receipt detail remains available through collection replay.
type CollectionReceiptSummary struct {
	ReceiptID             string
	AuditID               string
	ExecutionID           string
	RunID                 *string
	TerminalOutcome       TerminalOutcome
	TerminalRunGeneration *string
	TerminalRunSequence   *uint64
	Disposition           CollectionDisposition
	ErrorCode             *string
	RequestDigest         string
	CreatedAt             time.Time
}

type Event struct {
	AuditID        string
	Sequence       uint64
	Kind           string
	EntityID       string
	EntityRevision *uint64
	Summary        json.RawMessage
	CreatedAt      time.Time
}

type ReconcileSnapshot struct {
	Audit             Audit
	ReadyTasks        []Task
	HasUnsettledTasks bool
	// MaxConcurrentRuns and MaxConcurrentToolRuns are preflight hints from
	// Scheduler settings, the dispatch windows of the `model` and `tool`
	// classes. The CreateExecutionIntent SQL gate remains authoritative under
	// races.
	MaxConcurrentRuns     int
	MaxConcurrentToolRuns int

	Executions []Execution
	// RunStates is the lifecycle state of the Run bound to each execution in
	// Executions, keyed by Run ID and read in the same snapshot. A Run whose
	// row no longer exists has no entry.
	RunStates      map[string]runstore.WorkflowRunState
	Receipts       []CollectionReceiptSummary
	MoreExecutions bool
	MoreReceipts   bool
}

// DispatchWindowOpen reports whether the Audit may reserve another child Run
// of class: its outstanding Runs of that class are below the class's window
// (AUX-R3, AUX-R7). An unknown class or a window below one never opens.
func (s ReconcileSnapshot) DispatchWindowOpen(class resolved.LaneClass) bool {
	switch class {
	case resolved.LaneClassTool:
		return s.MaxConcurrentToolRuns >= 1 && s.Audit.OutstandingToolRunCount < s.MaxConcurrentToolRuns
	case resolved.LaneClassModel:
		return s.MaxConcurrentRuns >= 1 &&
			s.Audit.OutstandingRunCount-s.Audit.OutstandingToolRunCount < s.MaxConcurrentRuns
	default:
		return false
	}
}
