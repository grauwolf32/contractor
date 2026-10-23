package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func TestPostgresSchedulerSupervisorBoundsRealRunClaims(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	postgresStore := runstore.NewPostgresStore(pool)
	workflow := loadSchedulerWorkflow(t)
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	for _, runID := range []string{"run-lane-1", "run-lane-2", "run-lane-3"} {
		if _, err := runstoretest.CreateRun(ctx, postgresStore, runstore.CreateRunParams{
			LaneClass: resolved.LaneClassModel,
			RunID:     runID, OwnerID: "user-lanes", WorkflowName: workflow.Ref.Name,
			WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
			WorkflowSnapshot: workflowJSON, Parameters: map[string]string{"objective": "hold lane"},
			RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := postgresStore.TransitionRun(
			ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"},
		); err != nil {
			t.Fatal(err)
		}
		if _, err := postgresStore.TransitionRun(
			ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
		); err != nil {
			t.Fatal(err)
		}
	}
	settings := settingsstore.NewPostgresStore(pool)
	if _, err := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 2, ExpectedRevision: 1,
	}); err != nil {
		t.Fatal(err)
	}
	store := newPostgresLaneBlockingStore(postgresStore)
	scheduler, err := New(
		store, laneNoopPersistence{}, laneNoopArtifacts{}, laneNoopAllocator{},
		laneNoopWorkers{}, laneNoopPlanners{}, Options{
			PollInterval: 50 * time.Millisecond, ClaimDuration: time.Hour,
			PlannerSessions:  uncompletedSessions{},
			OperationTimeout: 2 * time.Second, PlannerTimeout: time.Second,
			FinalizationTimeout: time.Second, AbortTimeout: time.Second,
			LeaseScanInterval: time.Second,
			RuntimeTransport: RuntimeTransportSettings{
				ArtifactAPIURL: "https://control.test/private/v1", RequestTimeoutSeconds: 1,
			},
			Reports:  telemetry.NewRepository(pool),
			Settings: settings, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	runContext, stop := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { done <- scheduler.Run(runContext) }()

	first := receiveString(t, store.claimed, 5*time.Second, "first PostgreSQL claim")
	second := receiveString(t, store.claimed, 5*time.Second, "second PostgreSQL claim")
	if first == second {
		t.Fatalf("two lanes claimed the same Run %q", first)
	}
	assertNoString(t, store.claimed, 150*time.Millisecond, "third PostgreSQL claim exceeded limit two")
	store.unblock(first)
	third := receiveString(t, store.claimed, 5*time.Second, "replacement PostgreSQL claim")
	if third == first || third == second {
		t.Fatalf("replacement lane reclaimed prior Run %q", third)
	}
	store.unblock(second)
	store.unblock(third)
	store.waitReleased(t, 3, 5*time.Second)
	stop()
	if err := receiveError(t, done, 5*time.Second, "PostgreSQL Scheduler shutdown"); err != nil {
		t.Fatal(err)
	}
	if active, maximum := store.counts(); active != 0 || maximum != 2 {
		t.Fatalf("PostgreSQL lane counts active=%d maximum=%d, want 0/2", active, maximum)
	}
	for _, runID := range []string{"run-lane-1", "run-lane-2", "run-lane-3"} {
		run, err := postgresStore.GetRun(ctx, runID)
		if err != nil || run.State != runstore.RunFailed || run.SchedulerClaim != nil {
			t.Fatalf("Run %s after lane release = (%+v, %v)", runID, run, err)
		}
	}
}

func TestPostgresSchedulerSupervisorResizesAndDrainsDurableLanes(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	postgresStore := runstore.NewPostgresStore(pool)
	workflow := loadSchedulerWorkflow(t)
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	for _, runID := range []string{"run-resize-1", "run-resize-2", "run-resize-3", "run-resize-4"} {
		if _, err := runstoretest.CreateRun(ctx, postgresStore, runstore.CreateRunParams{
			LaneClass: resolved.LaneClassModel,
			RunID:     runID, OwnerID: "user-resize", WorkflowName: workflow.Ref.Name,
			WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
			WorkflowSnapshot: workflowJSON, Parameters: map[string]string{"objective": "hold lane"},
			RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := postgresStore.TransitionRun(
			ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"},
		); err != nil {
			t.Fatal(err)
		}
		if _, err := postgresStore.TransitionRun(
			ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
		); err != nil {
			t.Fatal(err)
		}
	}

	settings := settingsstore.NewPostgresStore(pool)
	store := newPostgresLaneBlockingStore(postgresStore)
	scheduler, err := New(
		store, laneNoopPersistence{}, laneNoopArtifacts{}, laneNoopAllocator{},
		laneNoopWorkers{}, laneNoopPlanners{}, Options{
			PollInterval: 60 * time.Millisecond, ClaimDuration: time.Hour,
			PlannerSessions:  uncompletedSessions{},
			OperationTimeout: 2 * time.Second, PlannerTimeout: time.Second,
			FinalizationTimeout: time.Second, AbortTimeout: time.Second,
			LeaseScanInterval: time.Second,
			RuntimeTransport: RuntimeTransportSettings{
				ArtifactAPIURL: "https://control.test/private/v1", RequestTimeoutSeconds: 1,
			},
			Reports:  telemetry.NewRepository(pool),
			Settings: settings, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	runContext, stop := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { done <- scheduler.Run(runContext) }()

	first := receiveString(t, store.claimed, 5*time.Second, "default serial PostgreSQL claim")
	assertNoString(t, store.claimed, 150*time.Millisecond, "migration default admitted a second lane")

	current, err := settings.GetSchedulerSettings(ctx)
	if err != nil || current.MaxConcurrentRuns != 1 {
		t.Fatalf("initial durable Scheduler settings = (%+v, %v)", current, err)
	}
	current, err = settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 2, ExpectedRevision: current.Revision,
	})
	if err != nil {
		t.Fatal(err)
	}
	scheduler.Wake()
	second := receiveString(t, store.claimed, 5*time.Second, "PostgreSQL claim after increase")
	if second == first {
		t.Fatalf("increased lane duplicated active Run %q", first)
	}

	current, err = settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 1, ExpectedRevision: current.Revision,
	})
	if err != nil {
		t.Fatal(err)
	}
	scheduler.Wake()
	store.unblock(first)
	store.waitReleased(t, 1, 5*time.Second)
	assertNoString(t, store.claimed, 180*time.Millisecond, "decrease replaced a drained lane above one")
	store.unblock(second)
	third := receiveString(t, store.claimed, 5*time.Second, "single PostgreSQL claim after drain")
	store.unblock(third)
	fourth := receiveString(t, store.claimed, 5*time.Second, "serial replacement after drain")
	store.unblock(fourth)
	store.waitReleased(t, 4, 5*time.Second)

	stop()
	if err := receiveError(t, done, 5*time.Second, "resized PostgreSQL Scheduler shutdown"); err != nil {
		t.Fatal(err)
	}
	if active, maximum := store.counts(); active != 0 || maximum != 2 {
		t.Fatalf("resized PostgreSQL lane counts active=%d maximum=%d, want 0/2", active, maximum)
	}
	persisted, err := settings.GetSchedulerSettings(ctx)
	if err != nil || persisted.MaxConcurrentRuns != 1 || persisted.Revision != current.Revision {
		t.Fatalf("persisted drained settings = (%+v, %v), want revision %d at one", persisted, err, current.Revision)
	}
}

type postgresLaneState struct {
	class       resolved.LaneClass
	gate        chan struct{}
	unblockOnce sync.Once
	once        sync.Once
	run         runstore.WorkflowRun
	err         error
	released    bool
}

type postgresLaneBlockingStore struct {
	*runstore.PostgresStore
	mu            sync.Mutex
	lanes         map[string]*postgresLaneState
	active        int
	maximum       int
	activeByClass map[resolved.LaneClass]int
	maxByClass    map[resolved.LaneClass]int
	releaseCount  int
	claimed       chan string
	released      chan int
}

func newPostgresLaneBlockingStore(store *runstore.PostgresStore) *postgresLaneBlockingStore {
	return &postgresLaneBlockingStore{
		PostgresStore: store, lanes: make(map[string]*postgresLaneState),
		activeByClass: make(map[resolved.LaneClass]int), maxByClass: make(map[resolved.LaneClass]int),
		claimed: make(chan string, 8), released: make(chan int, 8),
	}
}

func (s *postgresLaneBlockingStore) ClaimRunnableRun(
	ctx context.Context,
	claimID string,
	lease time.Duration,
	excludedRunIDs []string,
	laneClass resolved.LaneClass,
) (runstore.WorkflowRun, error) {
	run, err := s.PostgresStore.ClaimRunnableRun(ctx, claimID, lease, excludedRunIDs, laneClass)
	if err != nil {
		return run, err
	}
	if run.SchedulerLaneClass != laneClass {
		return run, fmt.Errorf("lane of class %s claimed Run %s of class %s", laneClass, run.RunID, run.SchedulerLaneClass)
	}
	s.mu.Lock()
	s.lanes[run.RunID] = &postgresLaneState{gate: make(chan struct{}), class: laneClass}
	s.active++
	if s.active > s.maximum {
		s.maximum = s.active
	}
	s.activeByClass[laneClass]++
	s.maxByClass[laneClass] = max(s.maxByClass[laneClass], s.activeByClass[laneClass])
	s.mu.Unlock()
	s.claimed <- run.RunID
	return run, nil
}

func (s *postgresLaneBlockingStore) GetRun(
	ctx context.Context,
	runID string,
) (runstore.WorkflowRun, error) {
	s.mu.Lock()
	lane := s.lanes[runID]
	s.mu.Unlock()
	if lane == nil {
		return s.PostgresStore.GetRun(ctx, runID)
	}
	select {
	case <-ctx.Done():
		return runstore.WorkflowRun{}, ctx.Err()
	case <-lane.gate:
	}
	lane.once.Do(func() {
		lane.run, lane.err = s.PostgresStore.TransitionRun(
			ctx, runID, runstore.RunRunning, runstore.RunFailed,
			runstore.Reason{Code: "supervisor_test_completed"},
		)
	})
	return lane.run, lane.err
}

// GetRunState is a progression attempt's first read: it holds the lane as
// GetRun does and reports the state of the Run GetRun completes.
func (s *postgresLaneBlockingStore) GetRunState(
	ctx context.Context,
	runID string,
) (runstore.WorkflowRunState, error) {
	s.mu.Lock()
	lane := s.lanes[runID]
	s.mu.Unlock()
	if lane == nil {
		return s.PostgresStore.GetRunState(ctx, runID)
	}
	run, err := s.GetRun(ctx, runID)
	return run.State, err
}

func (s *postgresLaneBlockingStore) ReleaseRunClaim(
	ctx context.Context,
	runID string,
	claimID string,
) error {
	err := s.PostgresStore.ReleaseRunClaim(ctx, runID, claimID)
	s.mu.Lock()
	if lane := s.lanes[runID]; lane != nil && !lane.released {
		lane.released = true
		s.active--
		s.activeByClass[lane.class]--
		s.releaseCount++
		s.released <- s.releaseCount
	}
	s.mu.Unlock()
	return err
}

func (s *postgresLaneBlockingStore) unblock(runID string) {
	s.mu.Lock()
	lane := s.lanes[runID]
	s.mu.Unlock()
	if lane == nil {
		return
	}
	lane.unblockOnce.Do(func() { close(lane.gate) })
}

func (s *postgresLaneBlockingStore) waitReleased(t *testing.T, count int, timeout time.Duration) {
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
		case <-s.released:
		case <-deadline.C:
			t.Fatalf("released %d PostgreSQL claims, want %d", current, count)
		}
	}
}

func (s *postgresLaneBlockingStore) counts() (int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active, s.maximum
}

// classMaximum is the largest number of claims lanes of class held at once.
func (s *postgresLaneBlockingStore) classMaximum(class resolved.LaneClass) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxByClass[class]
}

var _ Store = (*postgresLaneBlockingStore)(nil)
