// Package auditcontroller reconciles durable Audit state into ordinary pinned
// WorkflowRuns. It owns neither Workflow scheduling nor result acceptance.
package auditcontroller

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const (
	defaultPollInterval     = time.Second
	defaultClaimLease       = 30 * time.Second
	defaultOperationTimeout = 10 * time.Second
	defaultClaimBatch       = 8
	// A deletion attempt uses the Artifact cleanup budget under a renewed
	// lease twice as long, rather than the ordinary reconcile deadline.
	deletionClaimLease = 4 * time.Minute
	// maxDrainMutationsPerStep is the fixed number of drain mutations one
	// claimed reconcile step commits in its batched phase: terminal
	// observations, failures of unbound intents, child Run cancellation
	// requests or deletions of released terminal child Runs (LIM-AUX-14,
	// AUX-R97). It does not depend on the Audit's size and stays below
	// auditstore.MaxReconcileRows, the snapshot window its candidates come
	// from.
	maxDrainMutationsPerStep = 32
)

var (
	ErrInvalidSubmission = errors.New("Audit submission snapshot is invalid")
	ErrAlreadyRunning    = errors.New("Audit Controller is already running")
)

// Store is the claim-bound durable surface for controller claims, transitions,
// intents, observations, round acceptance, settlement and deletion: the claim
// seam of the Audit store. Collection and report finalization use the
// separate collector.
type Store interface {
	Claim(context.Context, auditstore.ClaimParams) ([]auditstore.ControllerClaim, error)
	ReleaseClaim(context.Context, auditstore.ControllerClaim) error
	RenewClaim(context.Context, auditstore.ControllerClaim, time.Duration) error
	TransitionClaimed(context.Context, auditstore.ClaimedTransitionParams) (auditstore.Audit, error)
	GetReconcileSnapshot(context.Context, auditstore.ControllerClaim) (auditstore.ReconcileSnapshot, error)
	ObserveTerminal(context.Context, auditstore.ObserveTerminalParams) (auditstore.Execution, error)
	ObserveSubmissionFailure(context.Context, auditstore.ObserveSubmissionFailureParams) (auditstore.Execution, error)
	SettleUndispatchedTasks(context.Context, auditstore.ControllerClaim, int) (int, error)
	SelectTasks(context.Context, auditstore.SelectTasksParams) (auditstore.Execution, bool, error)
	ReleaseDispatchHold(context.Context, auditstore.ControllerClaim) (auditstore.Audit, bool, error)
	NextLiveRunForDeletion(context.Context, auditstore.ControllerClaim) (string, bool, error)
	PurgeClaimed(context.Context, auditstore.ControllerClaim, string) error
}

// Reader is what the Controller reads of an Audit outside its reconcile
// snapshot: the exact members of an execution it submits. It is the read seam
// of the Audit store.
type Reader interface {
	ListExecutionTasks(context.Context, string, string) ([]auditstore.Task, error)
}

// RunStore is what the Controller reads and requests of its child Runs beyond
// the lifecycle states its reconcile snapshot carries.
type RunStore interface {
	GetRunEventCursor(context.Context, string) (runstore.WorkflowRunEventCursor, error)
	RequestRunCancellation(context.Context, string, runstore.WorkflowRunCancellation) (runstore.WorkflowRun, error)
}

type RunDeleter interface {
	DeleteReleasedTerminalRun(context.Context, string, string) error
}

type RunCreator interface {
	CreateAudit(context.Context, runservice.AuditCreateParams) (runservice.CreateResult, error)
}

type RunNotifier interface {
	Wake()
	Cancel(string)
}

type Collector interface {
	Collect(context.Context, auditstore.ControllerClaim, auditstore.ReconcileSnapshot, auditstore.Execution) (bool, error)
}

type Clock interface {
	Now() time.Time
	After(time.Duration) <-chan time.Time
}

type TaskIntake interface {
	AcceptContinuation(context.Context, auditstore.ControllerClaim, auditstore.ReconcileSnapshot) (auditstore.TaskIntakeResult, error)
	AcceptInitialTasks(context.Context, auditstore.ControllerClaim, auditstore.ReconcileSnapshot) (auditstore.TaskIntakeResult, error)
}

type Options struct {
	PollInterval     time.Duration
	ClaimLease       time.Duration
	OperationTimeout time.Duration
	ClaimBatch       int
	HolderID         string
	Clock            Clock
	NewID            func(string) (string, error)
	Logger           *slog.Logger
	Collector        Collector
	TaskIntake       TaskIntake
}

type Controller struct {
	store     Store
	reads     Reader
	runs      RunStore
	deleter   RunDeleter
	creator   RunCreator
	collector Collector
	notifier  RunNotifier
	options   Options
	// drainLimit is maxDrainMutationsPerStep, the bound of every batched
	// drain phase of a reconcile step.
	drainLimit int

	runMu   sync.Mutex
	running bool
	clockMu sync.Mutex
	idMu    sync.Mutex
}

func New(
	store Store,
	reads Reader,
	runs RunStore,
	deleter RunDeleter,
	creator RunCreator,
	notifier RunNotifier,
	options Options,
) (*Controller, error) {
	if store == nil || reads == nil || runs == nil || deleter == nil || creator == nil || notifier == nil {
		return nil, errors.New("Audit Controller dependencies are incomplete")
	}
	if options.PollInterval == 0 {
		options.PollInterval = defaultPollInterval
	}
	if options.ClaimLease == 0 {
		options.ClaimLease = defaultClaimLease
	}
	if options.OperationTimeout == 0 {
		options.OperationTimeout = defaultOperationTimeout
	}
	if options.ClaimBatch == 0 {
		options.ClaimBatch = defaultClaimBatch
	}
	if options.Clock == nil {
		options.Clock = realClock{}
	}
	if options.NewID == nil {
		options.NewID = randomid.New
	}
	if options.Logger == nil {
		options.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	if options.PollInterval <= 0 || options.OperationTimeout <= 0 ||
		options.ClaimLease < time.Second || options.ClaimLease > 5*time.Minute ||
		options.ClaimLease < 2*options.OperationTimeout ||
		options.ClaimBatch < 1 || options.ClaimBatch > auditstore.MaxClaimBatch {
		return nil, errors.New("Audit Controller bounds are invalid")
	}
	if options.HolderID == "" {
		value, err := options.NewID("audit-controller-")
		if err != nil {
			return nil, fmt.Errorf("generate Audit Controller identity: %w", err)
		}
		options.HolderID = value
	}
	return &Controller{
		store: store, reads: reads, runs: runs, deleter: deleter, creator: creator,
		collector: options.Collector,
		notifier:  notifier, options: options, drainLimit: maxDrainMutationsPerStep,
	}, nil
}

type realClock struct{}

func (realClock) Now() time.Time                                { return time.Now() }
func (realClock) After(duration time.Duration) <-chan time.Time { return time.After(duration) }
