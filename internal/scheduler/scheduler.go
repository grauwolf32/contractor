package scheduler

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"github.com/grauwolf32/contractor/internal/contracts"
	"io"
	"log/slog"
	"net/url"
	"sync"
	"time"
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
	active      map[string]context.CancelCauseFunc
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
	if err := validateRuntimeSettings(options.RuntimeSettings); err != nil {
		return nil, err
	}
	return &Scheduler{
		store: store, persistence: persistence, artifacts: artifactResolver,
		allocator: allocator, workers: workers, planners: planners, options: options,
		wake: make(chan struct{}, 1), active: make(map[string]context.CancelCauseFunc),
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
		options.NewID = schedulerID
	}
	if options.Logger == nil {
		options.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	options.TelemetrySecrets = append([]string(nil), options.TelemetrySecrets...)
}

func validateRuntimeSettings(settings contracts.RuntimeSettings) error {
	parsed, err := url.Parse(settings.ArtifactAPIURL)
	if err != nil || parsed.Host == "" || parsed.User != nil ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("Scheduler Artifact API URL is invalid")
	}
	if settings.RequestTimeoutSeconds <= 0 {
		return fmt.Errorf("Scheduler RuntimeSettings require a positive timeout")
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
	cancel := s.active[runID]
	s.activeMu.Unlock()
	if cancel != nil {
		cancel(cause)
	}
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

func schedulerID(prefix string) (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(buffer), nil
}
