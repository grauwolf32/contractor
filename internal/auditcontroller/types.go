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
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const (
	defaultPollInterval     = time.Second
	defaultClaimLease       = 30 * time.Second
	defaultOperationTimeout = 10 * time.Second
	defaultClaimBatch       = 8
)

var (
	ErrInvalidSubmission = errors.New("Audit submission snapshot is invalid")
	ErrAlreadyRunning    = errors.New("Audit Controller is already running")
)

// Store is the narrow claim-bound durable surface used by this delivery
// increment. Collection and settlement deliberately remain outside it.
type Store interface {
	Claim(context.Context, auditstore.ClaimParams) ([]auditstore.ControllerClaim, error)
	ReleaseClaim(context.Context, auditstore.ControllerClaim) error
	TransitionClaimed(context.Context, auditstore.ClaimedTransitionParams) (auditstore.Audit, error)
	TransitionRound(context.Context, auditstore.RoundTransitionParams) (auditstore.Round, error)
	GetReconcileSnapshot(context.Context, auditstore.ControllerClaim) (auditstore.ReconcileSnapshot, error)
	CreateExecutionIntent(context.Context, auditstore.CreateExecutionIntentParams) (auditstore.Execution, bool, error)
	NextItemAttempt(context.Context, auditstore.ControllerClaim, string) (int, error)
	ListExecutionItems(context.Context, string) ([]auditstore.ExecutionItem, error)
	ObserveTerminal(context.Context, auditstore.ObserveTerminalParams) (auditstore.Execution, error)
	ObserveSubmissionFailure(context.Context, auditstore.ObserveSubmissionFailureParams) (auditstore.Execution, error)
	AcceptNextRound(context.Context, auditstore.AcceptRoundParams) (auditstore.Round, bool, error)
	SettleUndispatched(context.Context, auditstore.ControllerClaim, int) (int, error)
	ReleaseDispatchHold(context.Context, auditstore.ControllerClaim) (auditstore.Audit, bool, error)
	NextLiveRunForDeletion(context.Context, auditstore.ControllerClaim) (string, bool, error)
	PurgeClaimed(context.Context, auditstore.ControllerClaim, string) error
}

type RunStore interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	GetRunEventCursor(context.Context, string) (runstore.WorkflowRunEventCursor, error)
	RequestRunCancellation(context.Context, string, runstore.WorkflowRunCancellation) (runstore.WorkflowRun, error)
	DeleteReleasedTerminalRun(context.Context, string, string) error
}

type RunCreator interface {
	CreateAudit(context.Context, runservice.AuditCreateParams) (runservice.CreateResult, error)
}

type RunNotifier interface {
	Wake()
	Cancel(string)
}

// PreparedSubmission contains the exact durable intent and the matching
// trusted Run Service request. Claim and Run identity callbacks are attached
// only by the Controller.
type PreparedSubmission struct {
	Intent auditstore.CreateExecutionIntentParams
	Run    runservice.AuditCreateParams
}

type SubmissionBuilder interface {
	Prepare(context.Context, auditstore.ReconcileSnapshot, auditstore.Item, int) (PreparedSubmission, error)
	PrepareRole(context.Context, auditstore.ReconcileSnapshot, string, int) (PreparedSubmission, error)
}

type Collector interface {
	Collect(context.Context, auditstore.ControllerClaim, auditstore.ReconcileSnapshot, auditstore.Execution) (bool, error)
	Finalize(context.Context, auditstore.ControllerClaim, auditstore.ReconcileSnapshot) (bool, error)
}

// RoundBuilder prepares the immutable manifest and exact proposal bindings for
// a later Audit round. The Store remains responsible for claim-bound atomic
// acceptance and consume-once validation.
type RoundBuilder interface {
	PrepareNextRound(
		context.Context,
		auditstore.ControllerClaim,
		auditstore.ReconcileSnapshot,
	) (auditstore.AcceptRoundParams, *auditstore.StopReason, error)
}

type Clock interface {
	Now() time.Time
	After(time.Duration) <-chan time.Time
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
	RoundBuilder     RoundBuilder
}

type Controller struct {
	store        Store
	runs         RunStore
	creator      RunCreator
	builder      SubmissionBuilder
	collector    Collector
	roundBuilder RoundBuilder
	notifier     RunNotifier
	options      Options
	wake         chan struct{}

	runMu   sync.Mutex
	running bool
	clockMu sync.Mutex
	idMu    sync.Mutex
}

func New(
	store Store,
	runs RunStore,
	creator RunCreator,
	builder SubmissionBuilder,
	notifier RunNotifier,
	options Options,
) (*Controller, error) {
	if store == nil || runs == nil || creator == nil || builder == nil || notifier == nil {
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
		options.NewID = randomID
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
		store: store, runs: runs, creator: creator, builder: builder,
		collector: options.Collector, roundBuilder: options.RoundBuilder,
		notifier: notifier, options: options, wake: make(chan struct{}, 1),
	}, nil
}

type realClock struct{}

func (realClock) Now() time.Time                                { return time.Now() }
func (realClock) After(duration time.Duration) <-chan time.Time { return time.After(duration) }
