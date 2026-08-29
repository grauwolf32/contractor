package scheduler

import (
	"context"
	"errors"
	"log/slog"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

var (
	ErrDeferred                 = errors.New("WorkflowRun execution is deferred")
	ErrClaimLost                = errors.New("WorkflowRun scheduler claim was lost")
	ErrRunCancellationRequested = errors.New("WorkflowRun cancellation was requested")
	ErrUnsupportedWorkflow      = errors.New("Workflow is outside the executable MVP shape")
)

// Store is the non-transactional durable state boundary used while no remote
// operation is in flight. AtomicPersistence owns the short multi-resource
// transactions used at lifecycle commit points.
type Store interface {
	ClaimRunnableRun(context.Context, string, time.Duration) (runstore.WorkflowRun, error)
	RenewRunClaim(context.Context, string, string, time.Duration) error
	ReleaseRunClaim(context.Context, string, string) error
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	TransitionRun(
		context.Context,
		string,
		runstore.WorkflowRunState,
		runstore.WorkflowRunState,
		runstore.Reason,
	) (runstore.WorkflowRun, error)
	ListStageExecutions(context.Context, string) ([]runstore.StageExecution, error)
	ListTerminalStageExecutionsWithAllocations(context.Context) ([]runstore.StageExecution, error)
	GetStageExecution(context.Context, string) (runstore.StageExecution, error)
	RecordStageAllocation(context.Context, runstore.StageAllocation) error
	ListStageAllocations(context.Context, string) ([]runstore.StageAllocation, error)
	RecordStageExecutionReport(context.Context, runstore.RecordStageExecutionReportParams) error
	EnterAborting(context.Context, runstore.EnterAbortingParams) error
}

type ContextPin struct {
	Name string
	Ref  contracts.ArtifactRef
}

type ResultAcceptance struct {
	RunID              string
	StageExecutionID   string
	Result             contracts.StageContentResult
	WorkflowOutputs    map[string]string
	OutputContracts    map[string]workflowconfig.ArtifactSlot
	ExpectedRunOutcome runstore.WorkflowRunState
}

// AtomicPersistence is deliberately high-level: implementations either commit
// every listed RunStore and ArtifactStore mutation or expose none of them.
type AtomicPersistence interface {
	CreateStageWithContext(
		context.Context,
		runstore.CreateStageExecutionParams,
		[]ContextPin,
	) (runstore.StageExecution, error)
	EnterFinalizingWithResult(context.Context, runstore.EnterFinalizingParams) error
	AcceptResultAndFinishRun(context.Context, ResultAcceptance) error
	AcceptResultDuringCancellation(
		context.Context,
		string,
		string,
		contracts.StageContentResult,
	) error
	CommitTerminationAndFinishRun(
		context.Context,
		string,
		string,
		runstore.WorkflowRunState,
		runstore.WorkflowRunState,
		runstore.Reason,
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
	ReserveAll(controlplane.ReservationRequest) ([]controlplane.Reservation, error)
	GetGrant(string) (controlplane.AllocationGrant, error)
	SetWriteFence(string) error
	Release(string) error
}

type WorkerController interface {
	PrepareAll(
		context.Context,
		[]controlplane.Reservation,
		contracts.RuntimeSettings,
	) (map[string]contracts.WorkerHandle, error)
	FinalizeAll(
		context.Context,
		[]controlplane.Reservation,
		string,
		time.Time,
	) (map[string]contracts.ExecutionReport, error)
	AbortAll(
		context.Context,
		[]controlplane.Reservation,
		string,
		contracts.TerminationError,
		time.Time,
	) (map[string]contracts.ExecutionReport, error)
	ReleaseAll(context.Context, []controlplane.Reservation) error
}

type PlannerRegistry interface {
	Create(string, planner.Invocation) (planner.Planner, error)
}

type Clock interface {
	Now() time.Time
	After(time.Duration) <-chan time.Time
}

type realClock struct{}

func (realClock) Now() time.Time                                { return time.Now() }
func (realClock) After(duration time.Duration) <-chan time.Time { return time.After(duration) }

type Options struct {
	PollInterval        time.Duration
	ClaimDuration       time.Duration
	OperationTimeout    time.Duration
	PlannerTimeout      time.Duration
	FinalizationTimeout time.Duration
	AbortTimeout        time.Duration
	RuntimeSettings     contracts.RuntimeSettings
	Clock               Clock
	NewID               func(string) (string, error)
	Logger              *slog.Logger
}
