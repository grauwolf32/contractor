package scheduler

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/randomid"
)

const (
	defaultPollInterval           = time.Second
	defaultClaimDuration          = 2 * time.Minute
	defaultOperationTimeout       = 15 * time.Second
	defaultPlannerTimeout         = 45 * time.Second
	defaultFinalizationTimeout    = 10 * time.Second
	defaultAbortTimeout           = 10 * time.Second
	defaultLeaseScanInterval      = time.Second
	defaultMetricsCleanupInterval = 24 * time.Hour
	defaultMetricsCleanupBatch    = 500
	metricsCleanupDrainPause      = time.Second
)

type Scheduler struct {
	store       Store
	persistence AtomicPersistence
	artifacts   ArtifactResolver
	allocator   Allocator
	workers     WorkerController
	planners    PlannerRegistry
	options     Options
	wake        chan struct{}
	activeMu    sync.Mutex
	active      map[string]activeRunClaim
	idMu        sync.Mutex
	clockMu     sync.Mutex
	runMu       sync.Mutex
	running     bool
	releaseMu   sync.Mutex
	releasing   map[string]struct{}
}

func New(
	store Store,
	persistence AtomicPersistence,
	artifactResolver ArtifactResolver,
	allocator Allocator,
	workers WorkerController,
	planners PlannerRegistry,
	options Options,
) (*Scheduler, error) {
	if store == nil || persistence == nil || artifactResolver == nil || allocator == nil ||
		workers == nil || planners == nil || options.Settings == nil {
		return nil, fmt.Errorf("Scheduler dependencies are incomplete")
	}
	applyOptionDefaults(&options)
	if options.PollInterval <= 0 || options.ClaimDuration <= 0 || options.OperationTimeout <= 0 ||
		options.PlannerTimeout <= 0 || options.FinalizationTimeout <= 0 || options.AbortTimeout <= 0 ||
		options.LeaseScanInterval <= 0 || options.MetricsCleanupInterval <= 0 ||
		options.MetricsCleanupBatch <= 0 || options.MetricsCleanupBatch > 10_000 {
		return nil, fmt.Errorf("Scheduler durations must be positive and cleanup batch must be at most 10000")
	}
	if err := validateRuntimeTransport(options.RuntimeTransport); err != nil {
		return nil, err
	}
	return &Scheduler{
		store: store, persistence: persistence, artifacts: artifactResolver,
		allocator: allocator, workers: workers, planners: planners, options: options,
		wake: make(chan struct{}, 1), active: make(map[string]activeRunClaim),
		releasing: make(map[string]struct{}),
	}, nil
}

func applyOptionDefaults(options *Options) {
	if options.PollInterval == 0 {
		options.PollInterval = defaultPollInterval
	}
	if options.ClaimDuration == 0 {
		options.ClaimDuration = defaultClaimDuration
	}
	if options.OperationTimeout == 0 {
		options.OperationTimeout = defaultOperationTimeout
	}
	if options.PlannerTimeout == 0 {
		options.PlannerTimeout = defaultPlannerTimeout
	}
	if options.FinalizationTimeout == 0 {
		options.FinalizationTimeout = defaultFinalizationTimeout
	}
	if options.AbortTimeout == 0 {
		options.AbortTimeout = defaultAbortTimeout
	}
	if options.LeaseScanInterval == 0 {
		options.LeaseScanInterval = defaultLeaseScanInterval
	}
	if options.MetricsCleanupInterval == 0 {
		options.MetricsCleanupInterval = defaultMetricsCleanupInterval
	}
	if options.MetricsCleanupBatch == 0 {
		options.MetricsCleanupBatch = defaultMetricsCleanupBatch
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
	options.TelemetrySecrets = append([]string(nil), options.TelemetrySecrets...)
}

func validateRuntimeTransport(settings RuntimeTransportSettings) error {
	parsed, err := url.Parse(settings.ArtifactAPIURL)
	if err != nil || parsed.Host == "" || parsed.User != nil ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("Scheduler Artifact API URL is invalid")
	}
	if settings.RequestTimeoutSeconds <= 0 {
		return fmt.Errorf("Scheduler Runtime transport requires a positive timeout")
	}
	return nil
}

// Wake requests an immediate claim attempt. It is edge-triggered and never
// blocks a public API handler.
func (s *Scheduler) Wake() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// Cancel interrupts an in-process Planner or lifecycle request after the
// public API has already made cancellation durable. A different Scheduler
// process observes the same state through claim reconciliation.
func (s *Scheduler) Cancel(runID string) {
	s.interruptRun(runID, ErrRunCancellationRequested)
	s.Wake()
}

func (s *Scheduler) interruptRun(runID string, cause error) {
	s.activeMu.Lock()
	current, ok := s.active[runID]
	s.activeMu.Unlock()
	if ok {
		current.cancel(cause)
	}
}

// activeRunClaim routes in-process interrupts to the lane holding one exact
// durable claim of a Run. stageExecutionID is the StageExecution that lane is
// currently progressing, so an allocation loss interrupts only its owning
// Stage and not a later Stage that a surviving owner has already advanced to.
type activeRunClaim struct {
	claimID          string
	stageExecutionID string
	cancel           context.CancelCauseFunc
}

func (s *Scheduler) registerActiveRun(runID, claimID string, cancel context.CancelCauseFunc) {
	s.activeMu.Lock()
	s.active[runID] = activeRunClaim{claimID: claimID, cancel: cancel}
	s.activeMu.Unlock()
}

// setActiveStage records the StageExecution the claim is currently
// progressing. It updates only the entry still owned by this claim so a
// released-and-reclaimed Run keeps the new lane's stage.
func (s *Scheduler) setActiveStage(runID, claimID, stageExecutionID string) {
	s.activeMu.Lock()
	if current, ok := s.active[runID]; ok && current.claimID == claimID {
		current.stageExecutionID = stageExecutionID
		s.active[runID] = current
	}
	s.activeMu.Unlock()
}

// interruptStageAllocationLoss cancels the lane only when the lost allocation
// belongs to the Stage that lane is currently executing. A loss for a Stage
// that already finished (its owner died after completion but before release)
// must not cancel the Run's current Stage; terminal release recovery reclaims
// the fenced grant instead. The loss, including its reason, is carried into
// the cancellation cause.
func (s *Scheduler) interruptStageAllocationLoss(loss controlplane.AllocationLoss) bool {
	s.activeMu.Lock()
	current, ok := s.active[loss.RunID]
	match := ok && current.stageExecutionID == loss.StageExecutionID
	s.activeMu.Unlock()
	if !match {
		return false
	}
	current.cancel(&AllocationLeaseLossError{Loss: loss})
	return true
}

// unregisterActiveRun removes only this claim's entry. After a claim is
// released another lane may already have claimed the Run and registered its
// own cancellation, which must stay reachable.
func (s *Scheduler) unregisterActiveRun(runID, claimID string) {
	s.activeMu.Lock()
	if current, ok := s.active[runID]; ok && current.claimID == claimID {
		delete(s.active, runID)
	}
	s.activeMu.Unlock()
}

func (s *Scheduler) newID(prefix string) (string, error) {
	s.idMu.Lock()
	defer s.idMu.Unlock()
	return s.options.NewID(prefix)
}

func (s *Scheduler) now() time.Time {
	s.clockMu.Lock()
	defer s.clockMu.Unlock()
	return s.options.Clock.Now()
}

func (s *Scheduler) after(duration time.Duration) <-chan time.Time {
	s.clockMu.Lock()
	defer s.clockMu.Unlock()
	return s.options.Clock.After(duration)
}

var (
	errControlPlaneStateLost      = errors.New("Control Plane state for durable allocations is unavailable")
	errControlPlaneAllocationLost = errors.New("Runtime Agent allocation is irreversibly lost")
)

// allocationLostError carries the specific loss reason so the Stage
// termination diagnostic reports it instead of collapsing every loss into one
// code. It unwraps to errControlPlaneAllocationLost so existing errors.Is
// checks keep matching.
type allocationLostError struct {
	reason controlplane.AllocationLossReason
}

func (e *allocationLostError) Error() string {
	if e.reason == "" {
		return errControlPlaneAllocationLost.Error()
	}
	return errControlPlaneAllocationLost.Error() + ": " + string(e.reason)
}

func (*allocationLostError) Unwrap() error { return errControlPlaneAllocationLost }
