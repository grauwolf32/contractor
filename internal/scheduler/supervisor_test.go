package scheduler

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

func TestSchedulerSupervisorBoundsDistinctRunLanes(t *testing.T) {
	store := newLaneTestStore("run-1", "run-2", "run-3")
	settings := newMutableSchedulerSettings(2, 500*time.Millisecond)
	scheduler := newLaneTestScheduler(t, store, settings, 500*time.Millisecond)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- scheduler.Run(ctx) }()

	first := receiveString(t, store.claimed, time.Second, "first claim")
	second := receiveString(t, store.claimed, time.Second, "second claim")
	if first == second {
		t.Fatalf("two lanes claimed the same Run %q", first)
	}
	assertNoString(t, store.claimed, 100*time.Millisecond, "third claim exceeded limit two")
	store.unblock(first)
	third := receiveString(t, store.claimed, time.Second, "third claim after one lane completed")
	if third == first || third == second {
		t.Fatalf("replacement lane claimed prior Run %q", third)
	}
	store.unblock(second)
	store.unblock(third)
	store.waitReleased(t, 3, time.Second)
	cancel()
	if err := receiveError(t, done, time.Second, "Scheduler shutdown"); err != nil {
		t.Fatal(err)
	}
	if active, maximum := store.counts(); active != 0 || maximum != 2 {
		t.Fatalf("lane counts active=%d maximum=%d, want 0/2", active, maximum)
	}
}

func TestSchedulerSupervisorResizesDrainsAndRecoversLostWake(t *testing.T) {
	poll := 60 * time.Millisecond
	store := newLaneTestStore("run-1", "run-2", "run-3", "run-4")
	settings := newMutableSchedulerSettings(1, poll)
	scheduler := newLaneTestScheduler(t, store, settings, poll)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- scheduler.Run(ctx) }()

	first := receiveString(t, store.claimed, time.Second, "initial serial claim")
	settings.drainReads()
	settings.set(2)
	scheduler.Wake()
	settings.waitRead(t, 2, time.Second)
	second := receiveString(t, store.claimed, time.Second, "claim after increase")

	settings.drainReads()
	settings.set(1)
	scheduler.Wake()
	settings.waitRead(t, 1, time.Second)
	store.unblock(first)
	store.waitReleased(t, 1, time.Second)
	assertNoString(t, store.claimed, 2*poll, "decrease started a replacement before drain")
	store.unblock(second)
	third := receiveString(t, store.claimed, time.Second, "single replacement after drain")

	settings.drainReads()
	settings.set(2)
	// Deliberately lose the process-local wake. Periodic refresh must still
	// observe the durable value and fill the second lane.
	settings.waitRead(t, 2, time.Second)
	fourth := receiveString(t, store.claimed, time.Second, "periodic increase without wake")
	store.unblock(third)
	store.unblock(fourth)
	store.waitReleased(t, 4, time.Second)
	cancel()
	if err := receiveError(t, done, time.Second, "Scheduler shutdown"); err != nil {
		t.Fatal(err)
	}
	if active, maximum := store.counts(); active != 0 || maximum > 2 {
		t.Fatalf("resized lane counts active=%d maximum=%d", active, maximum)
	}
}

func TestSchedulerSupervisorFailsClosedAndShutdownReleasesClaim(t *testing.T) {
	t.Run("settings unavailable before admission", func(t *testing.T) {
		store := newLaneTestStore("run-1")
		settings := newMutableSchedulerSettings(1, time.Second)
		settings.setError(errors.New("settings unavailable"))
		scheduler := newLaneTestScheduler(t, store, settings, 20*time.Millisecond)
		err := scheduler.Run(t.Context())
		if err == nil || store.claimCalls() != 0 {
			t.Fatalf("Run = %v, claims = %d", err, store.claimCalls())
		}
	})

	t.Run("cancelled process releases active claim", func(t *testing.T) {
		store := newLaneTestStore("run-1")
		settings := newMutableSchedulerSettings(1, time.Second)
		scheduler := newLaneTestScheduler(t, store, settings, 500*time.Millisecond)
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan error, 1)
		go func() { done <- scheduler.Run(ctx) }()
		_ = receiveString(t, store.claimed, time.Second, "active claim")
		cancel()
		if err := receiveError(t, done, time.Second, "cancelled Scheduler"); err != nil {
			t.Fatal(err)
		}
		if active, _ := store.counts(); active != 0 {
			t.Fatalf("shutdown leaked %d active claims", active)
		}
	})
}

func TestSchedulerSupervisorRotatesDeferredOwnerWithoutHotLoop(t *testing.T) {
	poll := 250 * time.Millisecond
	store := newLaneTestStore("paused-run-1", "paused-run-2", "runnable-run")
	store.deferRuns("paused-run-1", "paused-run-2")
	scheduler := newLaneTestScheduler(t, store, newMutableSchedulerSettings(2, poll), poll)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- scheduler.Run(ctx) }()

	for range 2 {
		claimed := receiveString(t, store.claimed, time.Second, "initial paused-owner claim")
		if claimed == "runnable-run" {
			t.Fatalf("runnable Run was unexpectedly ahead of paused fixtures: %q", claimed)
		}
	}
	// Deferred lanes must wait for poll/wake. Replacing them eagerly would let
	// an incompatible owner consume CPU and database claims in a hot loop.
	assertNoString(t, store.claimed, 80*time.Millisecond, "deferred lane hot-looped")

	var runnableClaimed bool
	for range 2 {
		if receiveString(t, store.claimed, time.Second, "rotated claim") == "runnable-run" {
			runnableClaimed = true
			break
		}
	}
	if !runnableClaimed {
		t.Fatal("deferred owner prevented runnable owner from progressing")
	}
	store.unblock("runnable-run")
	cancel()
	if err := receiveError(t, done, time.Second, "Scheduler shutdown after rotation"); err != nil {
		t.Fatal(err)
	}
	if active, maximum := store.counts(); active != 0 || maximum > 2 {
		t.Fatalf("rotating lane counts active=%d maximum=%d", active, maximum)
	}
}

func newLaneTestScheduler(
	t *testing.T,
	store *laneTestStore,
	settings SchedulerSettingsReader,
	poll time.Duration,
) *Scheduler {
	t.Helper()
	sequence := 0
	scheduler, err := New(
		store, laneNoopPersistence{}, laneNoopArtifacts{}, laneNoopAllocator{},
		laneNoopWorkers{}, laneNoopPlanners{}, Options{
			PollInterval: poll, ClaimDuration: time.Hour, OperationTimeout: 250 * time.Millisecond,
			PlannerTimeout: time.Second, FinalizationTimeout: time.Second, AbortTimeout: time.Second,
			LeaseScanInterval: time.Second, MetricsCleanupInterval: time.Hour,
			RuntimeSettings: contracts.RuntimeSettings{
				ArtifactAPIURL: "https://control.test/private/v1", RequestTimeoutSeconds: 1,
			},
			Settings: settings, Clock: &unsynchronizedLaneClock{now: time.Now().UTC()},
			NewID: func(prefix string) (string, error) {
				sequence++
				return fmt.Sprintf("%s%d", prefix, sequence), nil
			},
			Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return scheduler
}

// unsynchronizedLaneClock deliberately has mutable, unguarded state. The
// supervisor tests run it under the race detector to enforce Scheduler's
// synchronization boundary for injected clocks.
type unsynchronizedLaneClock struct {
	now   time.Time
	calls int
}

func (c *unsynchronizedLaneClock) Now() time.Time {
	c.calls++
	return c.now.Add(time.Duration(c.calls) * time.Microsecond)
}

func (c *unsynchronizedLaneClock) After(duration time.Duration) <-chan time.Time {
	c.calls++
	return time.After(duration)
}

type mutableSchedulerSettings struct {
	mu      sync.Mutex
	value   settingsstore.SchedulerSettings
	err     error
	reads   chan int
	advance time.Duration
}

func newMutableSchedulerSettings(value int, advance time.Duration) *mutableSchedulerSettings {
	return &mutableSchedulerSettings{
		value: settingsstore.SchedulerSettings{
			MaxConcurrentRuns: value, Revision: 1, UpdatedAt: time.Now().UTC(),
		},
		reads: make(chan int, 64), advance: advance,
	}
}

func (s *mutableSchedulerSettings) GetSchedulerSettings(ctx context.Context) (settingsstore.SchedulerSettings, error) {
	if err := ctx.Err(); err != nil {
		return settingsstore.SchedulerSettings{}, err
	}
	s.mu.Lock()
	value, err := s.value, s.err
	s.mu.Unlock()
	select {
	case s.reads <- value.MaxConcurrentRuns:
	default:
	}
	return value, err
}

func (s *mutableSchedulerSettings) set(value int) {
	s.mu.Lock()
	s.value.MaxConcurrentRuns = value
	s.value.Revision++
	s.value.UpdatedAt = s.value.UpdatedAt.Add(s.advance)
	s.err = nil
	s.mu.Unlock()
}

func (s *mutableSchedulerSettings) setError(err error) {
	s.mu.Lock()
	s.err = err
	s.mu.Unlock()
}

func (s *mutableSchedulerSettings) drainReads() {
	for {
		select {
		case <-s.reads:
		default:
			return
		}
	}
}

func (s *mutableSchedulerSettings) waitRead(t *testing.T, value int, timeout time.Duration) {
	t.Helper()
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	for {
		select {
		case read := <-s.reads:
			if read == value {
				return
			}
		case <-deadline.C:
			t.Fatalf("Scheduler did not read setting %d", value)
		}
	}
}

type laneTestStore struct {
	mu            sync.Mutex
	runs          []string
	started       map[string]bool
	claims        map[string]string
	gates         map[string]chan struct{}
	unblocked     map[string]bool
	deferred      map[string]bool
	active        int
	maximum       int
	claimCount    int
	releaseCount  int
	claimed       chan string
	releasedCount chan int
}

func newLaneTestStore(runIDs ...string) *laneTestStore {
	store := &laneTestStore{
		runs: append([]string(nil), runIDs...), started: make(map[string]bool),
		claims: make(map[string]string), gates: make(map[string]chan struct{}),
		unblocked: make(map[string]bool), deferred: make(map[string]bool),
		claimed:       make(chan string, len(runIDs)+8),
		releasedCount: make(chan int, len(runIDs)+8),
	}
	for _, runID := range runIDs {
		store.gates[runID] = make(chan struct{})
	}
	return store
}

func (s *laneTestStore) ClaimRunnableRun(
	ctx context.Context, claimID string, _ time.Duration,
) (runstore.WorkflowRun, error) {
	if err := ctx.Err(); err != nil {
		return runstore.WorkflowRun{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.claimCount++
	for _, runID := range s.runs {
		if s.started[runID] {
			continue
		}
		s.started[runID] = true
		s.claims[runID] = claimID
		s.active++
		if s.active > s.maximum {
			s.maximum = s.active
		}
		s.claimed <- runID
		ownerID := "runnable-owner"
		if s.deferred[runID] {
			ownerID = "paused-owner"
		}
		return runstore.WorkflowRun{RunID: runID, OwnerID: ownerID, State: runstore.RunRunning}, nil
	}
	return runstore.WorkflowRun{}, runstore.ErrNoWork
}

func (s *laneTestStore) RenewRunClaim(context.Context, string, string, time.Duration) error {
	return nil
}

func (s *laneTestStore) ReleaseRunClaim(_ context.Context, runID, claimID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.claims[runID] != claimID {
		return runstore.ErrConflict
	}
	delete(s.claims, runID)
	s.active--
	if s.deferred[runID] {
		s.started[runID] = false
		for index, candidate := range s.runs {
			if candidate == runID {
				s.runs = append(append(s.runs[:index:index], s.runs[index+1:]...), runID)
				break
			}
		}
	}
	s.releaseCount++
	s.releasedCount <- s.releaseCount
	return nil
}

func (s *laneTestStore) GetRun(ctx context.Context, runID string) (runstore.WorkflowRun, error) {
	s.mu.Lock()
	gate, exists := s.gates[runID]
	deferred := s.deferred[runID]
	s.mu.Unlock()
	if !exists {
		return runstore.WorkflowRun{}, runstore.ErrNotFound
	}
	if deferred {
		return runstore.WorkflowRun{}, runstore.ErrQueuePaused
	}
	select {
	case <-ctx.Done():
		return runstore.WorkflowRun{}, ctx.Err()
	case <-gate:
		return runstore.WorkflowRun{RunID: runID, State: runstore.RunSucceeded}, nil
	}
}

func (s *laneTestStore) deferRuns(runIDs ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, runID := range runIDs {
		s.deferred[runID] = true
	}
}

func (s *laneTestStore) TransitionRun(
	context.Context, string, runstore.WorkflowRunState, runstore.WorkflowRunState, runstore.Reason,
) (runstore.WorkflowRun, error) {
	return runstore.WorkflowRun{}, errors.New("unexpected Run transition")
}

func (s *laneTestStore) ListStageExecutions(context.Context, string) ([]runstore.StageExecution, error) {
	return nil, errors.New("unexpected Stage list")
}

func (s *laneTestStore) ListTerminalStageExecutionsWithAllocations(context.Context) ([]runstore.StageExecution, error) {
	return nil, nil
}

func (s *laneTestStore) GetStageExecution(context.Context, string) (runstore.StageExecution, error) {
	return runstore.StageExecution{}, errors.New("unexpected Stage read")
}

func (s *laneTestStore) RecordStageAllocation(context.Context, runstore.StageAllocation) error {
	return errors.New("unexpected allocation write")
}

func (s *laneTestStore) ListStageAllocations(context.Context, string) ([]runstore.StageAllocation, error) {
	return nil, nil
}

func (s *laneTestStore) MarkStageAllocationReleaseAttempt(context.Context, string) error {
	return errors.New("unexpected release attempt")
}

func (s *laneTestStore) MarkStageAllocationReleased(context.Context, string) error {
	return errors.New("unexpected release completion")
}

func (s *laneTestStore) RecordStageExecutionReport(context.Context, runstore.RecordStageExecutionReportParams) error {
	return errors.New("unexpected Stage report")
}

func (s *laneTestStore) RecordPlannerExecutionReport(context.Context, runstore.RecordPlannerExecutionReportParams) error {
	return errors.New("unexpected Planner report")
}

func (s *laneTestStore) RebuildStageMetrics(context.Context, string, string) error {
	return errors.New("unexpected metric rebuild")
}

func (s *laneTestStore) CleanupExpiredTelemetry(context.Context, time.Time, int) (int64, error) {
	return 0, nil
}

func (s *laneTestStore) unblock(runID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.unblocked[runID] {
		close(s.gates[runID])
		s.unblocked[runID] = true
	}
}

func (s *laneTestStore) waitReleased(t *testing.T, count int, timeout time.Duration) {
	t.Helper()
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	for {
		s.mu.Lock()
		current := s.releaseCount
		s.mu.Unlock()
		if current >= count {
			return
		}
		select {
		case <-s.releasedCount:
		case <-deadline.C:
			t.Fatalf("released %d Runs, want %d", current, count)
		}
	}
}

func (s *laneTestStore) counts() (int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active, s.maximum
}

func (s *laneTestStore) claimCalls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.claimCount
}

type laneNoopPersistence struct{}

func (laneNoopPersistence) CreateStageWithContext(context.Context, runstore.CreateStageExecutionParams, []ContextPin) (runstore.StageExecution, error) {
	return runstore.StageExecution{}, errors.New("unexpected persistence call")
}
func (laneNoopPersistence) EnterFinalizingWithResult(context.Context, runstore.EnterFinalizingParams) error {
	return errors.New("unexpected persistence call")
}
func (laneNoopPersistence) EnterAbortingWithTermination(context.Context, string, runstore.EnterAbortingParams) error {
	return errors.New("unexpected persistence call")
}
func (laneNoopPersistence) CommitResultProgression(context.Context, ResultProgression) error {
	return errors.New("unexpected persistence call")
}
func (laneNoopPersistence) AcceptResultDuringCancellation(context.Context, string, string, contracts.StageContentResult) error {
	return errors.New("unexpected persistence call")
}
func (laneNoopPersistence) CommitTerminationProgression(context.Context, TerminationProgression) error {
	return errors.New("unexpected persistence call")
}
func (laneNoopPersistence) CommitTerminationAndFinishRun(context.Context, string, string, runstore.WorkflowRunState, runstore.WorkflowRunState, runstore.Reason) error {
	return errors.New("unexpected persistence call")
}

type laneNoopArtifacts struct{}

func (laneNoopArtifacts) Resolve(context.Context, string, contracts.ArtifactRef) (ResolvedArtifact, error) {
	return ResolvedArtifact{}, errors.New("unexpected artifact resolution")
}

type laneNoopAllocator struct{}

func (laneNoopAllocator) ReserveAll(controlplane.ReservationRequest) ([]controlplane.Reservation, error) {
	return nil, errors.New("unexpected allocation")
}
func (laneNoopAllocator) GetGrant(string) (controlplane.AllocationGrant, error) {
	return controlplane.AllocationGrant{}, controlplane.ErrAllocationNotFound
}
func (laneNoopAllocator) GetReservation(string) (controlplane.Reservation, error) {
	return controlplane.Reservation{}, controlplane.ErrAllocationNotFound
}
func (laneNoopAllocator) SetWriteFence(string) error { return errors.New("unexpected fence") }
func (laneNoopAllocator) Release(string) error       { return errors.New("unexpected release") }
func (laneNoopAllocator) PollAllocationLosses() []controlplane.AllocationLoss {
	return nil
}

type laneNoopWorkers struct{}

func (laneNoopWorkers) PrepareAll(context.Context, []controlplane.Reservation, map[string]contracts.WorkerExecutionSettings) (map[string]contracts.WorkerHandle, error) {
	return nil, errors.New("unexpected Worker preparation")
}
func (laneNoopWorkers) FinalizeAll(context.Context, []controlplane.Reservation, string, time.Time) (map[string]contracts.AllocationFinalReport, error) {
	return nil, errors.New("unexpected Worker finalization")
}
func (laneNoopWorkers) AbortAll(context.Context, []controlplane.Reservation, string, contracts.TerminationError, time.Time) (map[string]contracts.AllocationFinalReport, error) {
	return nil, errors.New("unexpected Worker abort")
}
func (laneNoopWorkers) ReleaseAll(context.Context, []controlplane.Reservation) error {
	return errors.New("unexpected Worker release")
}

type laneNoopPlanners struct{}

func (laneNoopPlanners) Create(string, planner.Invocation) (planner.Planner, error) {
	return nil, errors.New("unexpected Planner creation")
}

func receiveString(t *testing.T, channel <-chan string, timeout time.Duration, operation string) string {
	t.Helper()
	select {
	case value := <-channel:
		return value
	case <-time.After(timeout):
		t.Fatalf("timeout waiting for %s", operation)
		return ""
	}
}

func assertNoString(t *testing.T, channel <-chan string, duration time.Duration, operation string) {
	t.Helper()
	select {
	case value := <-channel:
		t.Fatalf("%s: received %q", operation, value)
	case <-time.After(duration):
	}
}

func receiveError(t *testing.T, channel <-chan error, timeout time.Duration, operation string) error {
	t.Helper()
	select {
	case err := <-channel:
		return err
	case <-time.After(timeout):
		t.Fatalf("timeout waiting for %s", operation)
		return nil
	}
}
