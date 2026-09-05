package auditstore

import (
	"context"
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	MaxPageSize             = 200
	MaxClaimBatch           = 100
	MaxReconcileRows        = 200
	MaxSnapshotBytes        = 16 << 20
	MaxSummaryBytes         = 64 << 10
	MaxCollectionItems      = 64
	MaxArtifactLinksPerCall = 1024
	MaxArtifactRefBytes     = 4 << 10
	MaxCoverageArrayBytes   = 1 << 20
	MaxExecutionInputsBytes = 1 << 20
	MaxRoundPayloadBytes    = 64 << 20
	MaxExecutionIntentBytes = 64 << 20
	MaxCollectionBytes      = 16 << 20
	MaxRetainedRefsBytes    = 8 << 20
)

type AuditState string

const (
	AuditDraft         AuditState = "draft"
	AuditActive        AuditState = "active"
	AuditWaitingReview AuditState = "waiting_review"
	AuditPaused        AuditState = "paused"
	AuditFinalizing    AuditState = "finalizing"
	AuditCancelling    AuditState = "cancelling"
	AuditCompleted     AuditState = "completed"
	AuditCancelled     AuditState = "cancelled"
	AuditFailed        AuditState = "failed"
	AuditDeleting      AuditState = "deleting"
)

func (s AuditState) Valid() bool {
	switch s {
	case AuditDraft, AuditActive, AuditWaitingReview, AuditPaused,
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

type RoundState string

const (
	RoundProposed  RoundState = "proposed"
	RoundAccepted  RoundState = "accepted"
	RoundExecuting RoundState = "executing"
	RoundAssessing RoundState = "assessing"
	RoundClosed    RoundState = "closed"
)

func (s RoundState) Valid() bool {
	return s == RoundProposed || s == RoundAccepted || s == RoundExecuting ||
		s == RoundAssessing || s == RoundClosed
}

type ItemState string

const (
	ItemPending        ItemState = "pending"
	ItemAwaitingReview ItemState = "awaiting_review"
	ItemReady          ItemState = "ready"
	ItemSubmitted      ItemState = "submitted"
	ItemCollecting     ItemState = "collecting"
	ItemSettled        ItemState = "settled"
)

func (s ItemState) Valid() bool {
	return s == ItemPending || s == ItemAwaitingReview || s == ItemReady ||
		s == ItemSubmitted || s == ItemCollecting || s == ItemSettled
}

type ExecutionRole string

const (
	ExecutionDiscovery  ExecutionRole = "discovery"
	ExecutionCheck      ExecutionRole = "check"
	ExecutionAssessment ExecutionRole = "assessment"
)

func (r ExecutionRole) Valid() bool {
	return r == ExecutionDiscovery || r == ExecutionCheck || r == ExecutionAssessment
}

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
)

func (d CollectionDisposition) Valid() bool {
	return d == CollectionAccepted || d == CollectionMissingOutput ||
		d == CollectionInvalidResult || d == CollectionExecutionFailed ||
		d == CollectionExecutionCancelled
}

type FinalDisposition string

const (
	FinalAccepted           FinalDisposition = "accepted-result"
	FinalMissingOutput      FinalDisposition = "missing-output"
	FinalInvalidResult      FinalDisposition = "invalid-result"
	FinalExecutionFailed    FinalDisposition = "execution-failed"
	FinalExecutionCancelled FinalDisposition = "execution-cancelled"
	FinalExcluded           FinalDisposition = "excluded"
	FinalNotApplicable      FinalDisposition = "not-applicable"
)

func (d FinalDisposition) Valid() bool {
	switch d {
	case FinalAccepted, FinalMissingOutput, FinalInvalidResult,
		FinalExecutionFailed, FinalExecutionCancelled, FinalExcluded,
		FinalNotApplicable:
		return true
	default:
		return false
	}
}

type CoverageStatus string

const (
	CoverageNotTested     CoverageStatus = "not-tested"
	CoverageInconclusive  CoverageStatus = "inconclusive"
	CoverageSatisfied     CoverageStatus = "satisfied"
	CoverageViolated      CoverageStatus = "violated"
	CoverageNotApplicable CoverageStatus = "not-applicable"
	CoverageBlocked       CoverageStatus = "blocked"
	CoverageExcluded      CoverageStatus = "excluded"
)

func (s CoverageStatus) Valid() bool {
	switch s {
	case CoverageNotTested, CoverageInconclusive, CoverageSatisfied,
		CoverageViolated, CoverageNotApplicable, CoverageBlocked, CoverageExcluded:
		return true
	default:
		return false
	}
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
	AuditID               string
	OwnerID               string
	ProjectID             string
	Profile               ProfileIdentity
	ProfileSnapshot       json.RawMessage
	InputSelection        json.RawMessage
	BaselineSnapshot      json.RawMessage
	State                 AuditState
	Revision              uint64
	CurrentRoundID        *string
	Dispatch              DispatchState
	Hold                  HoldState
	DeadlineAt            *time.Time
	Limits                Limits
	ReservedRunCount      int
	SubmittedRunCount     int
	OutstandingRunCount   int
	RetainedEvidenceBytes int64
	EventSequence         uint64
	StopReason            *StopReason
	CreatedAt             time.Time
	UpdatedAt             time.Time
	StartedAt             *time.Time
	FinishedAt            *time.Time
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
	OwnerID         string
	ProjectID       *string
	State           *AuditState
	ProfileName     *string
	ProfileVersion  *string
	BeforeCreatedAt *time.Time
	BeforeAuditID   string
	Limit           int
}

type ListItemsParams struct {
	OwnerID           string
	AuditID           string
	RoundID           *string
	State             *ItemState
	SubjectKey        *string
	AfterRoundOrdinal *int
	AfterItemOrdinal  *int
	AfterItemID       string
	Limit             int
}

type MutationOperation string

const (
	MutationCreate MutationOperation = "audit.create"
	MutationStart  MutationOperation = "audit.start"
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

type ClaimedTransitionParams struct {
	Claim            ControllerClaim
	ExpectedRevision uint64
	ExpectedState    AuditState
	TargetState      AuditState
	Reason           *StopReason
}

type RoundTransitionParams struct {
	Claim            ControllerClaim
	RoundID          string
	ExpectedRevision uint64
	ExpectedState    RoundState
	TargetState      RoundState
}

type ExactArtifact struct {
	Ref       contracts.ArtifactRef `json:"ref"`
	Digest    string                `json:"digest"`
	MediaType string                `json:"mediaType,omitempty"`
	SizeBytes int64                 `json:"sizeBytes,omitempty"`
}

type MaterializedItem struct {
	ItemID       string
	ItemKey      string
	Ordinal      int
	Kind         string
	SubjectKey   string
	Task         ExactArtifact
	WorkflowRole string
	InitialState ItemState
	Coverage     Coverage
}

type Coverage struct {
	Status    CoverageStatus `json:"status"`
	Requested []string       `json:"requested"`
	Completed []string       `json:"completed"`
	Gaps      []string       `json:"gaps"`
	Rationale string         `json:"rationale,omitempty"`
}

type MaterializeRoundParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	RoundID          string
	RoundOrdinal     int
	Manifest         ExactArtifact
	BaselineSnapshot json.RawMessage
	DeadlineAt       time.Time
	Items            []MaterializedItem
	IdempotencyKey   string
	RequestDigest    string
}

type Round struct {
	RoundID           string
	AuditID           string
	Ordinal           int
	Manifest          ExactArtifact
	State             RoundState
	ExpectedItemCount int
	Revision          uint64
	CreatedAt         time.Time
	UpdatedAt         time.Time
}

type Item struct {
	ItemID              string
	AuditID             string
	RoundID             string
	ItemKey             string
	Ordinal             int
	Kind                string
	SubjectKey          string
	Task                ExactArtifact
	WorkflowRole        string
	State               ItemState
	FinalDisposition    *FinalDisposition
	AcceptedResult      *ExactArtifact
	LastExecutionItemID *string
	CreatedAt           time.Time
	UpdatedAt           time.Time
}

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

type ExecutionMemberIntent struct {
	ExecutionItemID string
	ItemID          string
	BatchOrdinal    int
	ItemAttempt     int
	Task            ExactArtifact
	Inputs          []ExactArtifact
}

type CreateExecutionIntentParams struct {
	Claim         ControllerClaim
	ExecutionID   string
	RoundID       *string
	Role          ExecutionRole
	RoleAttempt   *int
	Manifest      ExactArtifact
	SubmissionKey string
	RequestDigest string
	Members       []ExecutionMemberIntent
}

type Execution struct {
	ExecutionID           string
	AuditID               string
	RoundID               *string
	Role                  ExecutionRole
	RoleAttempt           *int
	Manifest              ExactArtifact
	SubmissionKey         string
	RequestDigest         string
	RunID                 *string
	State                 ExecutionState
	TerminalOutcome       *TerminalOutcome
	TerminalRunGeneration *string
	TerminalRunSequence   *uint64
	TerminalObservedAt    *time.Time
	CreatedAt             time.Time
	UpdatedAt             time.Time
}

type ExecutionItem struct {
	ExecutionItemID       string
	ExecutionID           string
	AuditID               string
	RoundID               string
	ItemID                string
	BatchOrdinal          int
	ItemAttempt           int
	Task                  ExactArtifact
	Inputs                []ExactArtifact
	State                 ItemState
	CollectionDisposition *CollectionDisposition
	Result                *ExactArtifact
	CreatedAt             time.Time
	CollectedAt           *time.Time
}

type BindRunParams struct {
	Claim       ControllerClaim
	ExecutionID string
	RunID       string
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

type CollectionItem struct {
	ExecutionItemID  string
	Disposition      CollectionDisposition
	Result           *ExactArtifact
	Retryable        bool
	FinalDisposition FinalDisposition
	Coverage         Coverage
}

type ArtifactLink struct {
	LogicalKey       string          `json:"logicalKey"`
	Artifact         ExactArtifact   `json:"artifact"`
	SourceProvenance json.RawMessage `json:"sourceProvenance"`
	DisplayRef       string          `json:"displayRef,omitempty"`
}

type CollectParams struct {
	Claim         ControllerClaim
	ReceiptID     string
	ExecutionID   string
	Disposition   CollectionDisposition
	SourceOutput  *ExactArtifact
	Retained      []ArtifactLink
	ErrorCode     *string
	RequestDigest string
	Items         []CollectionItem
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

type CoverageRow struct {
	AuditID    string
	RoundID    string
	ItemID     string
	Ordinal    int
	ItemKey    string
	SubjectKey string
	Coverage   Coverage
	Result     *ExactArtifact
	UpdatedAt  time.Time
}

type ReconcileSnapshot struct {
	Audit          Audit
	Round          *Round
	Items          []Item
	Executions     []Execution
	Receipts       []CollectionReceiptSummary
	MoreItems      bool
	MoreExecutions bool
	MoreReceipts   bool
}

// OwnerRepository never accepts an untrusted owner in stored payloads.
type OwnerRepository interface {
	LookupMutationReplay(context.Context, string, MutationOperation, string, string) (Audit, bool, error)
	CreateDraft(context.Context, CreateDraftParams) (Audit, bool, error)
	Get(context.Context, string, string) (Audit, error)
	List(context.Context, ListParams) ([]Audit, error)
	ListItemsPage(context.Context, ListItemsParams) ([]Item, error)
	Transition(context.Context, TransitionParams) (Audit, bool, error)
	MaterializeRound(context.Context, MaterializeRoundParams) (Audit, bool, error)
}

// ControllerRepository is a separate trusted surface. Every mutating call is
// bound to one live claim epoch; owner labels or Run metadata never authorize it.
type ControllerRepository interface {
	Claim(context.Context, ClaimParams) ([]ControllerClaim, error)
	RenewClaim(context.Context, ControllerClaim, time.Duration) (ControllerClaim, error)
	ReleaseClaim(context.Context, ControllerClaim) error
	TransitionClaimed(context.Context, ClaimedTransitionParams) (Audit, error)
	TransitionRound(context.Context, RoundTransitionParams) (Round, error)
	CreateExecutionIntent(context.Context, CreateExecutionIntentParams) (Execution, bool, error)
	BindRun(context.Context, BindRunParams) (Execution, error)
	ObserveTerminal(context.Context, ObserveTerminalParams) (Execution, error)
	ObserveSubmissionFailure(context.Context, ObserveSubmissionFailureParams) (Execution, error)
	Collect(context.Context, CollectParams) (CollectionReceipt, bool, error)
	GetReconcileSnapshot(context.Context, ControllerClaim) (ReconcileSnapshot, error)
}
