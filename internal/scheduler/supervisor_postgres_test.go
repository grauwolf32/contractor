package scheduler

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

// postgresSupervisorPoll is the poll interval of the supervisor PostgreSQL
// tests. Their manual clock fires it only when a test completes a fill round.
const postgresSupervisorPoll = time.Hour

// newPostgresSupervisor returns a Scheduler whose lanes use store and whose
// polls happen when the test fires clock. Each store operation may take up to
// a minute, so the test's own deadline, not a short budget a loaded host
// could exceed, bounds the test (AGENTS.md 3.5).
func newPostgresSupervisor(
	t *testing.T, pool *pgxpool.Pool, store Store, settings SchedulerSettingsReader, clock Clock,
) *Scheduler {
	t.Helper()
	scheduler, err := New(
		store, laneNoopPersistence{}, laneNoopArtifacts{}, laneNoopAllocator{},
		laneNoopWorkers{}, laneNoopPlanners{}, Options{
			PollInterval: postgresSupervisorPoll, ClaimDuration: time.Hour, Clock: clock,
			PlannerSessions:  uncompletedSessions{},
			OperationTimeout: time.Minute, PlannerTimeout: time.Second,
			FinalizationTimeout: time.Second, AbortTimeout: time.Second,
			LeaseScanInterval: time.Minute,
			RuntimeTransport: RuntimeTransportSettings{
				ArtifactAPIURL: "https://control.test", RequestTimeoutSeconds: 1,
			},
			Reports:  telemetry.NewRepository(pool),
			Settings: settings, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return scheduler
}

// roundSettings counts the supervisor's settings reads. Every fill round
// opens with one, and the supervisor handles one round at a time.
type roundSettings struct {
	SchedulerSettingsReader
	mu      sync.Mutex
	reads   int
	changed chan struct{}
}

func newRoundSettings(settings SchedulerSettingsReader) *roundSettings {
	return &roundSettings{SchedulerSettingsReader: settings, changed: make(chan struct{})}
}

func (s *roundSettings) GetSchedulerSettings(ctx context.Context) (settingsstore.SchedulerSettings, error) {
	value, err := s.SchedulerSettingsReader.GetSchedulerSettings(ctx)
	s.mu.Lock()
	s.reads++
	close(s.changed)
	s.changed = make(chan struct{})
	s.mu.Unlock()
	return value, err
}

func (s *roundSettings) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reads
}

// awaitReads waits until the supervisor has read its settings count times.
func (s *roundSettings) awaitReads(t *testing.T, ctx context.Context, count int) {
	t.Helper()
	for {
		s.mu.Lock()
		reads, changed := s.reads, s.changed
		s.mu.Unlock()
		if reads >= count {
			return
		}
		select {
		case <-changed:
		case <-ctx.Done():
			t.Fatalf("Scheduler read its settings %d times, want %d: %v", reads, count, context.Cause(ctx))
		}
	}
}

// completePollRound fires the next poll and the one after it, waits until
// the supervisor read its settings for the second, and then until every lane
// started so far has attempted its claim. The second round's timer is armed
// only while the supervisor handles the first, so by then the first round has
// started every lane it decided to start, and the claim of such a lane has
// reached store.claimed: the absence checks wait for these events instead of
// a fixed window. The terminal release monitor arms the second timer of each
// poll.
func completePollRound(
	t *testing.T, ctx context.Context, clock *manualPollClock, settings *roundSettings, store *postgresLaneBlockingStore,
) {
	t.Helper()
	reads := settings.count()
	for range 2 {
		clock.fireWhenArmedBefore(t, ctx, postgresSupervisorPoll, 2)
	}
	settings.awaitReads(t, ctx, reads+2)
	store.awaitLaneClaims(t, ctx)
}

// assertNoClaim fails when a lane claimed a Run since the test last received
// from store.claimed. Callers complete the fill round that would start the
// lane first.
func assertNoClaim(t *testing.T, store *postgresLaneBlockingStore, operation string) {
	t.Helper()
	select {
	case runID := <-store.claimed:
		t.Fatalf("%s: claimed %q", operation, runID)
	default:
	}
}

// With two lanes, the supervisor holds at most two real PostgreSQL claims: a
// completed fill round starts no third lane while both are held, and a freed
// lane claims the remaining Run at once.
func TestPostgresSchedulerSupervisorBoundsRealRunClaims(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	postgresStore := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	workflow := loadSchedulerWorkflow(t)
	for _, runID := range []string{"run-lane-1", "run-lane-2", "run-lane-3"} {
		createRunningPoolRun(t, ctx, postgresStore, runID, workflow, map[string]string{"objective": "hold lane"})
	}
	settings := settingsstore.NewPostgresStore(pool)
	if _, err := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 2, ExpectedRevision: 1,
	}); err != nil {
		t.Fatal(err)
	}
	store := newPostgresLaneBlockingStore(postgresStore)
	rounds := newRoundSettings(settings)
	clock := newManualPollClock()
	scheduler := newPostgresSupervisor(t, pool, store, rounds, clock)
	store.observeLaneStarts(scheduler)
	runContext, stop := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { done <- scheduler.Run(runContext) }()

	first := awaitValue(t, ctx, store.claimed, "first PostgreSQL claim")
	second := awaitValue(t, ctx, store.claimed, "second PostgreSQL claim")
	if first == second {
		t.Fatalf("two lanes claimed the same Run %q", first)
	}
	completePollRound(t, ctx, clock, rounds, store)
	assertNoClaim(t, store, "third PostgreSQL claim exceeded limit two")
	store.unblock(first)
	third := awaitValue(t, ctx, store.claimed, "replacement PostgreSQL claim")
	if third == first || third == second {
		t.Fatalf("replacement lane reclaimed prior Run %q", third)
	}
	store.unblock(second)
	store.unblock(third)
	store.waitReleased(t, ctx, 3)
	stop()
	if err := awaitValue(t, ctx, done, "PostgreSQL Scheduler shutdown"); err != nil {
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

// The migration's single model lane admits one Run; raising the durable
// limit starts a second lane, and lowering it lets a drained lane end without
// a replacement until the pool is below the new limit.
func TestPostgresSchedulerSupervisorResizesAndDrainsDurableLanes(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	postgresStore := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	workflow := loadSchedulerWorkflow(t)
	for _, runID := range []string{"run-resize-1", "run-resize-2", "run-resize-3", "run-resize-4"} {
		createRunningPoolRun(t, ctx, postgresStore, runID, workflow, map[string]string{"objective": "hold lane"})
	}

	settings := settingsstore.NewPostgresStore(pool)
	store := newPostgresLaneBlockingStore(postgresStore)
	rounds := newRoundSettings(settings)
	clock := newManualPollClock()
	scheduler := newPostgresSupervisor(t, pool, store, rounds, clock)
	store.observeLaneStarts(scheduler)
	runContext, stop := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { done <- scheduler.Run(runContext) }()

	first := awaitValue(t, ctx, store.claimed, "default serial PostgreSQL claim")
	completePollRound(t, ctx, clock, rounds, store)
	assertNoClaim(t, store, "migration default admitted a second lane")

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
	second := awaitValue(t, ctx, store.claimed, "PostgreSQL claim after increase")
	if second == first {
		t.Fatalf("increased lane duplicated active Run %q", first)
	}

	current, err = settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 1, ExpectedRevision: current.Revision,
	})
	if err != nil {
		t.Fatal(err)
	}
	// The supervisor reads the decrease before the first lane drains, so the
	// drained lane ends against the new limit.
	reads := rounds.count()
	scheduler.Wake()
	rounds.awaitReads(t, ctx, reads+1)
	store.unblock(first)
	store.waitReleased(t, ctx, 1)
	completePollRound(t, ctx, clock, rounds, store)
	assertNoClaim(t, store, "decrease replaced a drained lane above one")
	store.unblock(second)
	third := awaitValue(t, ctx, store.claimed, "single PostgreSQL claim after drain")
	store.unblock(third)
	fourth := awaitValue(t, ctx, store.claimed, "serial replacement after drain")
	store.unblock(fourth)
	store.waitReleased(t, ctx, 4)

	stop()
	if err := awaitValue(t, ctx, done, "resized PostgreSQL Scheduler shutdown"); err != nil {
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
	// starts and attempts count the lanes a Scheduler started and their
	// completed claim attempts; progress closes when either changes.
	starts   int
	attempts int
	progress chan struct{}
}

func newPostgresLaneBlockingStore(store *runstore.PostgresStore) *postgresLaneBlockingStore {
	return &postgresLaneBlockingStore{
		PostgresStore: store, lanes: make(map[string]*postgresLaneState),
		activeByClass: make(map[resolved.LaneClass]int), maxByClass: make(map[resolved.LaneClass]int),
		claimed: make(chan string, 8), released: make(chan int, 8), progress: make(chan struct{}),
	}
}

// observeLaneStarts counts each lane scheduler starts: a lane's first act is
// to generate its claim ID.
func (s *postgresLaneBlockingStore) observeLaneStarts(scheduler *Scheduler) {
	next := scheduler.options.NewID
	scheduler.options.NewID = func(prefix string) (string, error) {
		if prefix == "claim_" {
			s.mu.Lock()
			s.starts++
			s.progressedLocked()
			s.mu.Unlock()
		}
		return next(prefix)
	}
}

func (s *postgresLaneBlockingStore) progressedLocked() {
	close(s.progress)
	s.progress = make(chan struct{})
}

// awaitLaneClaims waits until every lane started so far has attempted its
// claim.
func (s *postgresLaneBlockingStore) awaitLaneClaims(t *testing.T, ctx context.Context) {
	t.Helper()
	for {
		s.mu.Lock()
		starts, attempts, progress := s.starts, s.attempts, s.progress
		s.mu.Unlock()
		if attempts >= starts {
			return
		}
		select {
		case <-progress:
		case <-ctx.Done():
			t.Fatalf("%d of %d started lanes attempted their claim: %v", attempts, starts, context.Cause(ctx))
		}
	}
}

func (s *postgresLaneBlockingStore) ClaimRunnableRun(
	ctx context.Context,
	claimID string,
	lease time.Duration,
	excludedRunIDs []string,
	laneClass resolved.LaneClass,
) (runstore.WorkflowRun, error) {
	// The attempt counts once its claim, if any, is on s.claimed.
	defer func() {
		s.mu.Lock()
		s.attempts++
		s.progressedLocked()
		s.mu.Unlock()
	}()
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

// waitReleased waits until lanes released count claims before ctx, the
// test's own deadline, ends.
func (s *postgresLaneBlockingStore) waitReleased(t *testing.T, ctx context.Context, count int) {
	t.Helper()
	for {
		s.mu.Lock()
		current := s.releaseCount
		s.mu.Unlock()
		if current >= count {
			return
		}
		select {
		case <-s.released:
		case <-ctx.Done():
			t.Fatalf("released %d PostgreSQL claims, want %d: %v", current, count, context.Cause(ctx))
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
