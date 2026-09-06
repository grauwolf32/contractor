package auditstore

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	MaxPageSize                    = 200
	MaxClaimBatch                  = 100
	MaxReconcileRows               = 200
	MaxSnapshotBytes               = 16 << 20
	MaxSummaryBytes                = 64 << 10
	MaxCollectionItems             = 64
	MaxArtifactLinksPerCall        = 1024
	MaxArtifactRefBytes            = 4 << 10
	MaxCoverageArrayBytes          = 1 << 20
	MaxExecutionInputsBytes        = 1 << 20
	MaxRoundPayloadBytes           = 64 << 20
	MaxExecutionIntentBytes        = 64 << 20
	MaxCollectionBytes             = 16 << 20
	MaxRetainedRefsBytes           = 8 << 20
	MaxAttemptProjection           = MaxPageSize * 10
	MaxAuditRoleExecutionsPerRound = 256
	ItemOriginSchema               = "contractor.audit.item-origin.v1"
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
	CollectionContractInvalid    CollectionDisposition = "collection-contract-invalid"
)

func (d CollectionDisposition) Valid() bool {
	return d == CollectionAccepted || d == CollectionMissingOutput ||
		d == CollectionInvalidResult || d == CollectionExecutionFailed ||
		d == CollectionExecutionCancelled || d == CollectionContractInvalid
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
	CoverageNotTested      CoverageStatus = "not-tested"
	CoverageInconclusive   CoverageStatus = "inconclusive"
	CoverageSatisfied      CoverageStatus = "satisfied"
	CoverageViolated       CoverageStatus = "violated"
	CoverageNotApplicable  CoverageStatus = "not-applicable"
	CoverageBlocked        CoverageStatus = "blocked"
	CoverageExcluded       CoverageStatus = "excluded"
	CoverageTracedComplete CoverageStatus = "traced-complete"
	CoverageTracedPartial  CoverageStatus = "traced-partial"
	CoverageUnmapped       CoverageStatus = "unmapped"
)

func (s CoverageStatus) Valid() bool {
	switch s {
	case CoverageNotTested, CoverageInconclusive, CoverageSatisfied,
		CoverageViolated, CoverageNotApplicable, CoverageBlocked, CoverageExcluded,
		CoverageTracedComplete, CoverageTracedPartial, CoverageUnmapped:
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
	MutationCreate     MutationOperation = "audit.create"
	MutationStart      MutationOperation = "audit.start"
	MutationTransition MutationOperation = "audit.transition"
	MutationDelete     MutationOperation = "audit.delete"
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

// ItemOrigin is the immutable attribution of a materialized Audit item to the
// exact source document and normalized inventory entry from which it was
// generated. ProvenanceIncomplete is reserved for rows created before this
// projection existed; newly materialized items must always be complete.
type ItemOrigin struct {
	Schema                   string                 `json:"schema"`
	SourceRef                *contracts.ArtifactRef `json:"sourceRef,omitempty"`
	SourceContentDigest      string                 `json:"sourceContentDigest,omitempty"`
	SourceMediaType          string                 `json:"sourceMediaType,omitempty"`
	CanonicalInventoryDigest string                 `json:"canonicalInventoryDigest,omitempty"`
	EntryKey                 string                 `json:"entryKey"`
	EntryVersion             string                 `json:"entryVersion,omitempty"`
	ProvenanceIncomplete     bool                   `json:"provenanceIncomplete,omitempty"`
}

type MaterializedItem struct {
	ItemID       string
	ItemKey      string
	Ordinal      int
	Kind         string
	SubjectKey   string
	Task         ExactArtifact
	Origin       ItemOrigin
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
	Origin              ItemOrigin
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
	WorkflowRole  string
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
	WorkflowRole          string
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
	RunProvenance         *RunProvenance
	RunDeletedAt          *time.Time
	CreatedAt             time.Time
	UpdatedAt             time.Time
}

// RunProvenance is a bounded, non-secret tombstone captured when a child Run
// is bound. It deliberately stores configuration identity and a closure digest
// rather than the resolved Workflow body, credential material, or live grants.
type RunProvenance struct {
	Schema               string              `json:"schema"`
	RunID                string              `json:"runId"`
	Workflow             *WorkflowProvenance `json:"workflow,omitempty"`
	ProvenanceIncomplete bool                `json:"provenanceIncomplete,omitempty"`
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

// RunCreationIntent is the claim-bound immutable authority consumed by the
// trusted Run Service. It contains no secret material and is never exposed by
// the owner-facing repository or public API.
type RunCreationIntent struct {
	OwnerID   string
	ProjectID string
	Execution Execution
	Items     []ExecutionItem
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
	ExecutionItemID     string
	Disposition         CollectionDisposition
	Result              *ExactArtifact
	Retryable           bool
	FinalDisposition    FinalDisposition
	Coverage            Coverage
	FindingAssociations []FindingAssociation
}

// FindingAssociation is trusted importer output. It binds one exact intake
// receipt to the accepted result for this precise execution item; Workers
// cannot write these rows through the generic Artifact API.
type FindingAssociation struct {
	AssessmentID       string
	ReceiptID          string
	Proposal           ExactArtifact
	SemanticAssessment string
}

type ArtifactLink struct {
	LogicalKey       string          `json:"logicalKey"`
	Artifact         ExactArtifact   `json:"artifact"`
	SourceProvenance json.RawMessage `json:"sourceProvenance"`
	DisplayRef       string          `json:"displayRef,omitempty"`
	CreatedAt        time.Time       `json:"createdAt,omitempty"`
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

type ItemAttempt struct {
	ExecutionItemID       string                 `json:"executionItemId"`
	ExecutionID           string                 `json:"executionId"`
	ItemID                string                 `json:"itemId"`
	ItemAttempt           int                    `json:"itemAttempt"`
	Role                  ExecutionRole          `json:"role"`
	State                 ItemState              `json:"state"`
	CollectionDisposition *CollectionDisposition `json:"collectionDisposition,omitempty"`
	Result                *ExactArtifact         `json:"result,omitempty"`
	TerminalOutcome       *TerminalOutcome       `json:"terminalOutcome,omitempty"`
	RunID                 *string                `json:"runId,omitempty"`
	RunDeleted            bool                   `json:"runDeleted"`
	RunProvenance         *RunProvenance         `json:"runProvenance,omitempty"`
	CreatedAt             time.Time              `json:"createdAt"`
	CollectedAt           *time.Time             `json:"collectedAt,omitempty"`
}

type CollectionDispositionCounts struct {
	AcceptedResult     int `json:"acceptedResult"`
	MissingOutput      int `json:"missingOutput"`
	InvalidResult      int `json:"invalidResult"`
	ExecutionFailed    int `json:"executionFailed"`
	ExecutionCancelled int `json:"executionCancelled"`
	ContractInvalid    int `json:"contractInvalid"`
}

// ReportFinding is the trusted, bounded finding projection consumed while an
// immutable Audit report is assembled. The proposal document remains in the
// Artifact plane; this row pins its exact retained version and the exact
// assessment/analyst decision (if any) selected by this report revision.
type ReportFinding struct {
	FindingID         string
	State             string
	FirstProposal     ExactArtifact
	DuplicateTargetID *string
	Assessment        *ReportFindingAssessment
	Decision          *ReportFindingDecision
	Revision          uint64
}

type ReportFindingAssessment struct {
	AssessmentID       string
	SemanticAssessment string
	Result             ExactArtifact
	DirectVerification bool
	Contract           *ExactArtifact
	AcceptedAt         time.Time
}

type ReportFindingDecision struct {
	DecisionID      string
	ActorID         string
	Verdict         string
	Severity        *string
	Rationale       string
	SubjectRevision uint64
	SubjectDigest   string
	CreatedAt       time.Time
}

type CommitReportParams struct {
	Claim                 ControllerClaim
	ExpectedAuditRevision uint64
	RoundID               string
	ExpectedRoundRevision uint64
	Machine               ArtifactLink
	Summary               ArtifactLink
	RequestDigest         string
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
	RoleExecutions []Execution
	Receipts       []CollectionReceiptSummary
	RoleReceipts   []CollectionReceiptSummary
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
	ListItemAttempts(context.Context, string, string, []string) (map[string][]ItemAttempt, error)
	Transition(context.Context, TransitionParams) (Audit, bool, error)
	RequestDelete(context.Context, DeleteParams) (Audit, bool, error)
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
	NextItemAttempt(context.Context, ControllerClaim, string) (int, error)
	ListExecutionItems(context.Context, string) ([]ExecutionItem, error)
	GetRunCreationIntent(context.Context, ControllerClaim, string) (RunCreationIntent, error)
	BindRun(context.Context, BindRunParams) (Execution, error)
	ObserveTerminal(context.Context, ObserveTerminalParams) (Execution, error)
	ObserveSubmissionFailure(context.Context, ObserveSubmissionFailureParams) (Execution, error)
	Collect(context.Context, CollectParams) (CollectionReceipt, bool, error)
	CommitReport(context.Context, CommitReportParams) (Audit, error)
	SettleUndispatched(context.Context, ControllerClaim, int) (int, error)
	ReleaseDispatchHold(context.Context, ControllerClaim) (Audit, bool, error)
	NextLiveRunForDeletion(context.Context, ControllerClaim) (string, bool, error)
	PurgeClaimed(context.Context, ControllerClaim, string) error
	GetReconcileSnapshot(context.Context, ControllerClaim) (ReconcileSnapshot, error)
	GetArtifactLink(context.Context, string, string) (ArtifactLink, error)
}

// RoleOutputLogicalKey is the stable Audit-owned address of one accepted
// output from a named discovery or assessment role in an immutable Round.
func RoleOutputLogicalKey(roundOrdinal int, workflowRole, logicalOutput string) string {
	return fmt.Sprintf("round/%d/role/%s/output/%s", roundOrdinal, workflowRole, logicalOutput)
}
