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
	"github.com/grauwolf32/contractor/internal/randomid"
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
	ExpireReportReview(context.Context, auditstore.ControllerClaim, uint64) (bool, error)
	RenewExpiredItemReview(context.Context, auditstore.ControllerClaim, uint64) (bool, error)
	TransitionClaimed(context.Context, auditstore.ClaimedTransitionParams) (auditstore.Audit, error)
	TransitionRound(context.Context, auditstore.RoundTransitionParams) (auditstore.Round, error)
	GetReconcileSnapshot(context.Context, auditstore.ControllerClaim) (auditstore.ReconcileSnapshot, error)
	CompletePreparation(context.Context, auditstore.ControllerClaim, uint64) (auditstore.Audit, error)
	AcceptInitialRound(context.Context, auditstore.AcceptInitialRoundParams) (auditstore.Round, bool, error)
	CreateExecutionIntent(context.Context, auditstore.CreateExecutionIntentParams) (auditstore.Execution, bool, error)
	NextItemAttempts(context.Context, auditstore.ControllerClaim, []string) (map[string]int, error)
	ObserveTerminal(context.Context, auditstore.ObserveTerminalParams) (auditstore.Execution, error)
	ObserveSubmissionFailure(context.Context, auditstore.ObserveSubmissionFailureParams) (auditstore.Execution, error)
	AcceptNextRound(context.Context, auditstore.AcceptRoundParams) (auditstore.Round, bool, error)
	SettleUndispatched(context.Context, auditstore.ControllerClaim, int) (int, error)
	ReleaseDispatchHold(context.Context, auditstore.ControllerClaim) (auditstore.Audit, bool, error)
	NextLiveRunForDeletion(context.Context, auditstore.ControllerClaim) (string, bool, error)
	PurgeClaimed(context.Context, auditstore.ControllerClaim, string) error
}

// Reader is what the Controller reads of an Audit outside its reconcile
// snapshot: the exact members of an execution it submits. It is the read seam
// of the Audit store.
type Reader interface {
	ListExecutionItems(context.Context, string) ([]auditstore.ExecutionItem, error)
	ListItemsByIDs(context.Context, string, []string) ([]auditstore.Item, error)
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

// PreparedSubmission contains the exact durable intent and the matching
// trusted Run Service request. Claim and Run identity callbacks are attached
// only by the Controller.
type PreparedSubmission struct {
	Intent auditstore.CreateExecutionIntentParams
	Run    runservice.AuditCreateParams
}

// CheckExecutionMember is one independently attempted AuditItem selected for
// an ordered check execution. Batch position is derived from slice order; it
// is never supplied by a model or by mutable Workflow state.
type CheckExecutionMember struct {
	Item    auditstore.Item
	Attempt int
}

type SubmissionBuilder interface {
	WithClaim(auditstore.ControllerClaim) SubmissionBuilder
	// PrepareBatch may select a non-empty ordered prefix of the supplied
	// candidates to satisfy trusted encoded-package byte limits.
	PrepareBatch(context.Context, auditstore.ReconcileSnapshot, []CheckExecutionMember) (PreparedSubmission, error)
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

type InitialRoundBuilder interface {
	PrepareInitialRound(context.Context, auditstore.ControllerClaim, auditstore.ReconcileSnapshot) (auditstore.AcceptInitialRoundParams, *auditstore.StopReason, error)
}

type Clock interface {
	Now() time.Time
	After(time.Duration) <-chan time.Time
}

type Options struct {
	PollInterval        time.Duration
	ClaimLease          time.Duration
	OperationTimeout    time.Duration
	ClaimBatch          int
	HolderID            string
	Clock               Clock
	NewID               func(string) (string, error)
	Logger              *slog.Logger
	Collector           Collector
	RoundBuilder        RoundBuilder
	InitialRoundBuilder InitialRoundBuilder
}

type Controller struct {
	store               Store
	reads               Reader
	runs                RunStore
	deleter             RunDeleter
	creator             RunCreator
	builder             SubmissionBuilder
	collector           Collector
	roundBuilder        RoundBuilder
	initialRoundBuilder InitialRoundBuilder
	notifier            RunNotifier
	options             Options

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
	builder SubmissionBuilder,
	notifier RunNotifier,
	options Options,
) (*Controller, error) {
	if store == nil || reads == nil || runs == nil || deleter == nil || creator == nil || builder == nil || notifier == nil {
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
		store: store, reads: reads, runs: runs, deleter: deleter, creator: creator, builder: builder,
		collector: options.Collector, roundBuilder: options.RoundBuilder, initialRoundBuilder: options.InitialRoundBuilder,
		notifier: notifier, options: options,
	}, nil
}

type realClock struct{}

func (realClock) Now() time.Time                                { return time.Now() }
func (realClock) After(duration time.Duration) <-chan time.Time { return time.After(duration) }
