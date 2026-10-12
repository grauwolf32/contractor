package scheduler

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

var (
	ErrDeferred                 = errors.New("WorkflowRun execution is deferred")
	ErrClaimLost                = errors.New("WorkflowRun scheduler claim was lost")
	ErrRunCancellationRequested = errors.New("WorkflowRun cancellation was requested")
	ErrAllocationLeaseLost      = errors.New("Runtime Agent allocation lease was lost")
	ErrUnsupportedWorkflow      = errors.New("Workflow is outside the executable MVP shape")
)

type AllocationLeaseLossError struct {
	Loss controlplane.AllocationLoss
}

func (e *AllocationLeaseLossError) Error() string {
	return "allocation " + e.Loss.AllocationID + " was lost: " + string(e.Loss.Reason)
}

func (*AllocationLeaseLossError) Unwrap() error { return ErrAllocationLeaseLost }

// Store is the non-transactional durable state boundary used while no remote
// operation is in flight. AtomicPersistence owns the short multi-resource
// transactions used at lifecycle commit points.
type Store interface {
	ClaimRunnableRun(context.Context, string, time.Duration) (runstore.WorkflowRun, error)
	RenewRunClaim(context.Context, string, string, time.Duration) (runstore.WorkflowRunState, error)
	ReleaseRunClaim(context.Context, string, string) error
	DeferRunClaim(context.Context, string, string) error
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	// GetRunState reads only the Run's lifecycle state; decisions on the
	// state alone use it instead of GetRun.
	GetRunState(context.Context, string) (runstore.WorkflowRunState, error)
	GetOwnerQueueControl(context.Context, string) (runstore.OwnerQueueControl, error)
	TransitionRun(
		context.Context,
		string,
		runstore.WorkflowRunState,
		runstore.WorkflowRunState,
		runstore.Reason,
	) (runstore.WorkflowRun, error)
	// GetActiveStageExecution reads the Run's one active StageExecution:
	// runstore.ErrNotFound when it has none, runstore.ErrMultipleActiveStages
	// when it has more (SCH-R114). A progression attempt reads only this
	// Stage, not the Run's Stage history.
	GetActiveStageExecution(context.Context, string) (runstore.StageExecution, error)
	ListStageExecutions(context.Context, string) ([]runstore.StageExecution, error)
	ListTerminalStageExecutionsWithAllocations(context.Context) ([]runstore.StageExecution, error)
	GetStageExecution(context.Context, string) (runstore.StageExecution, error)
	ListStageAllocations(context.Context, string) ([]runstore.StageAllocation, error)
	MarkStageAllocationReleaseAttempt(context.Context, string) error
	MarkStageAllocationReleased(context.Context, string) error
	// RecordStageOutputPublicationFailure counts a terminal commit of the
	// finalizing StageExecution (runID, stageExecutionID) that a transient
	// Project output publication failure aborted (PRJ-R58).
	RecordStageOutputPublicationFailure(context.Context, string, string) error
}

// ReportStore keeps the bounded execution reports of a StageExecution and
// rebuilds its StageMetrics from them; telemetry.Repository implements it.
type ReportStore interface {
	RecordAllocationReport(context.Context, telemetry.AllocationReportEnvelope) error
	RecordPlannerReport(context.Context, telemetry.PlannerReportEnvelope) error
	RebuildStageMetrics(context.Context, string, string) (telemetry.StageMetricsRecord, error)
}

type ContextPin struct {
	Name string
	Ref  contracts.ArtifactRef
}

type NextStageCreation struct {
	Params      runstore.CreateStageExecutionParams
	ContextPins []ContextPin
}

// StageProgression is the complete post-Stage policy decision. NextStage is
// present exactly for next/retry/escalate; TerminalRunState is present exactly
// for succeed/fail. Persistence commits the decision together with the target
// StageExecution or terminal WorkflowRun transition.
type StageProgression struct {
	Decision         runstore.RecordStageTransitionDecisionParams
	NextStage        *NextStageCreation
	TerminalRunState runstore.WorkflowRunState
	RunReason        runstore.Reason
}

type ResultProgression struct {
	RunID            string
	StageExecutionID string
	Result           contracts.StageContentResult
	WorkflowOutputs  map[string]string
	OutputContracts  map[string]resolved.ArtifactSlot
	Progression      StageProgression
}

type TerminationProgression struct {
	RunID            string
	StageExecutionID string
	Progression      StageProgression
}

// AtomicPersistence is deliberately high-level: implementations either commit
// every listed RunStore and ArtifactStore mutation or expose none of them.
type AtomicPersistence interface {
	AdmitStage(context.Context, string, string) (runstore.StageExecution, error)
	// DiscardUnadmittedPlacement deletes the durable allocation rows of a
	// pending Run's never admitted Stage and, in the same transaction, runs
	// discard to drop the matching live reservation; a discard error rolls
	// the deletion back.
	DiscardUnadmittedPlacement(ctx context.Context, runID, stageID string, discard func() error) error
	CreateStageWithContext(
		context.Context,
		runstore.CreateStageExecutionParams,
		[]ContextPin,
	) (runstore.StageExecution, error)
	EnterFinalizingWithResult(context.Context, runstore.EnterFinalizingParams) error
	EnterAbortingWithTermination(
		context.Context,
		string,
		runstore.EnterAbortingParams,
	) error
	CommitResultProgression(context.Context, ResultProgression) error
	AcceptResultDuringCancellation(
		context.Context,
		string,
		string,
		contracts.StageContentResult,
	) error
	CommitTerminationProgression(context.Context, TerminationProgression) error
	CommitTerminationAndFinishRun(
		context.Context,
		string,
		string,
		runstore.WorkflowRunState,
		runstore.WorkflowRunState,
		runstore.Reason,
	) error
	FailRunWithActiveStages(
		context.Context,
		string,
		runstore.WorkflowRunState,
		runstore.WorkflowRunState,
		runstore.Reason,
		runstore.StageTermination,
		string,
	) error
}

type ResolvedArtifact struct {
	Ref       contracts.ArtifactRef
	MediaType string
}

type ArtifactResolver interface {
	Resolve(context.Context, string, contracts.ArtifactRef) (ResolvedArtifact, error)
}

type Allocator interface {
	ReserveAllContext(context.Context, controlplane.ReservationRequest) ([]controlplane.Reservation, error)
	GetGrant(string) (controlplane.AllocationGrant, error)
	GetReservation(string) (controlplane.Reservation, error)
	SetWriteFence(string) error
	// DiscardUnprepared drops a Stage's committed reservation that was never
	// prepared, leaving no released tombstone, so the Stage can be placed again.
	DiscardUnprepared(string) error
	Release(string) error
	PollAllocationLosses() []controlplane.AllocationLoss
}

// WorkerController owns the Runtime batch budgets: PrepareAll bounds one
// prepare batch and ReleaseAll one release batch by their own deadlines, while
// FinalizeAll and AbortAll honour the stored terminal deadline. Callers pass
// cancellation only and never wrap a batch in the store-operation budget.
type WorkerController interface {
	PrepareAll(
		context.Context,
		[]controlplane.Reservation,
		map[string]runtimesettings.WorkerExecutionSettings,
	) (map[string]contracts.WorkerHandle, error)
	FinalizeAll(
		context.Context,
		[]controlplane.Reservation,
		string,
		time.Time,
	) (map[string]reporting.AllocationFinalReport, error)
	AbortAll(
		context.Context,
		[]controlplane.Reservation,
		string,
		contracts.TerminationError,
		time.Time,
	) (map[string]reporting.AllocationFinalReport, error)
	ReleaseAll(context.Context, []controlplane.Reservation) error
}

type CredentialResolver interface {
	ResolveLLMCredential(
		context.Context,
		llmgateway.LLMCredentialRef,
		llmgateway.LLMGatewayConfigRef,
	) (contracts.SecretString, error)
}

type RuntimeCredentialResolver interface {
	UsePlaintext(
		context.Context,
		string,
		[]contracts.RuntimeCredentialKind,
		func(contracts.RuntimeCredentialKind, []byte) error,
	) error
}

type PlannerRegistry interface {
	Create(string, planner.Invocation) (planner.Planner, error)
}

type PlannerTelemetryRegistry interface {
	Create(string, telemetry.PlannerAdapterSettings) (telemetry.PlannerTelemetry, error)
}

type RunSkillInitializer interface {
	InitializeRunSkills(context.Context, string) (runstore.WorkflowRun, error)
}

// SchedulerSettingsReader is the PostgreSQL-authoritative desired lane limit.
// Implementations must be safe for periodic reads while Operations mutates the
// same resource.
type SchedulerSettingsReader interface {
	GetSchedulerSettings(context.Context) (settingsstore.SchedulerSettings, error)
}

// Clock implementations may be injected without their own synchronization;
// Scheduler serializes method calls shared by lanes and maintenance loops.
type Clock interface {
	Now() time.Time
	After(time.Duration) <-chan time.Time
}

type realClock struct{}

func (realClock) Now() time.Time                                { return time.Now() }
func (realClock) After(duration time.Duration) <-chan time.Time { return time.After(duration) }

// Options carries the resolved ServerConfig budgets; each duration below
// names its ServerConfig path and is required (OPS-R15).
type Options struct {
	GatewayRecovery *gatewayrecovery.Service
	// PollInterval is scheduler.pollInterval: the lane refill and terminal
	// release recovery period, and the retry period after a failed claim
	// renewal.
	PollInterval time.Duration
	// ClaimDuration is scheduler.claimDuration, the Run claim lease, renewed
	// every third of it.
	ClaimDuration time.Duration
	// OperationTimeout is scheduler.operationTimeout: one store operation
	// (claim, renewal, read, transition or commit). Runtime batches carry
	// their own budgets in the WorkerController.
	OperationTimeout time.Duration
	// PlannerTimeout is plannerTimeout: the Stage deadline from admission.
	PlannerTimeout      time.Duration
	FinalizationTimeout time.Duration
	AbortTimeout        time.Duration
	// LeaseScanInterval is scheduler.leaseScanInterval, the period of the
	// allocation-loss scan.
	LeaseScanInterval  time.Duration
	RuntimeTransport   RuntimeTransportSettings
	Credentials        CredentialResolver
	RuntimeCredentials RuntimeCredentialResolver
	PlannerTelemetry   PlannerTelemetryRegistry
	Reports            ReportStore
	RunSkills          RunSkillInitializer
	Settings           SchedulerSettingsReader
	TelemetrySecrets   []string
	Clock              Clock
	NewID              func(string) (string, error)
	Logger             *slog.Logger
}

// RuntimeTransportSettings holds the static endpoints needed when materializing
// per-allocation Runtime settings; credentials are resolved from pinned config.
type RuntimeTransportSettings struct {
	ArtifactAPIURL        string
	RequestTimeoutSeconds int
}
