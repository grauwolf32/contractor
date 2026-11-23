package scheduler

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// manualPollClock never fires a timer on its own: a test fires every armed
// timer of one duration and waits for timers to be armed, so polls happen
// exactly when the test says.
type manualPollClock struct {
	mu      sync.Mutex
	now     time.Time
	timers  map[time.Duration][]chan time.Time
	changed chan struct{}
}

func newManualPollClock() *manualPollClock {
	return &manualPollClock{
		now: time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC), timers: make(map[time.Duration][]chan time.Time),
		changed: make(chan struct{}),
	}
}

func (c *manualPollClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *manualPollClock) After(duration time.Duration) <-chan time.Time {
	tick := make(chan time.Time, 1)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.timers[duration] = append(c.timers[duration], tick)
	close(c.changed)
	c.changed = make(chan struct{})
	return tick
}

// fireWhenArmed waits until armed timers of duration are pending, then fires
// all of them.
func (c *manualPollClock) fireWhenArmed(t *testing.T, duration time.Duration, armed int) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	c.fireWhenArmedBefore(t, ctx, duration, armed)
}

// fireWhenArmedBefore is fireWhenArmed within ctx, the test's own deadline.
func (c *manualPollClock) fireWhenArmedBefore(t *testing.T, ctx context.Context, duration time.Duration, armed int) {
	t.Helper()
	for {
		c.mu.Lock()
		pending := c.timers[duration]
		if len(pending) >= armed {
			delete(c.timers, duration)
			now := c.now
			c.mu.Unlock()
			for _, tick := range pending {
				tick <- now
			}
			return
		}
		changed := c.changed
		c.mu.Unlock()
		select {
		case <-changed:
		case <-ctx.Done():
			t.Fatalf("%d timers of %v armed, want %d: %v", len(pending), duration, armed, context.Cause(ctx))
		}
	}
}

// claimAttemptStore records every claim attempt of a laneTestStore, with or
// without work, by lane class.
type claimAttemptStore struct {
	*laneTestStore
	mu        sync.Mutex
	attempts  map[resolved.LaneClass]int
	attempted chan resolved.LaneClass
}

func newClaimAttemptStore(store *laneTestStore) *claimAttemptStore {
	return &claimAttemptStore{
		laneTestStore: store, attempts: make(map[resolved.LaneClass]int),
		attempted: make(chan resolved.LaneClass, 64),
	}
}

func (s *claimAttemptStore) ClaimRunnableRun(
	ctx context.Context, claimID string, lease time.Duration, excluded []string, class resolved.LaneClass,
) (runstore.WorkflowRun, error) {
	run, err := s.laneTestStore.ClaimRunnableRun(ctx, claimID, lease, excluded, class)
	s.mu.Lock()
	s.attempts[class]++
	s.mu.Unlock()
	s.attempted <- class
	return run, err
}

func (s *claimAttemptStore) attemptCounts() map[resolved.LaneClass]int {
	s.mu.Lock()
	defer s.mu.Unlock()
	counts := make(map[resolved.LaneClass]int, len(s.attempts))
	for class, count := range s.attempts {
		counts[class] = count
	}
	return counts
}

// releaseTotal is the number of claims lanes have released.
func (s *laneTestStore) releaseTotal() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.releaseCount
}

func receiveClass(t *testing.T, channel <-chan resolved.LaneClass, operation string) resolved.LaneClass {
	t.Helper()
	select {
	case class := <-channel:
		return class
	case <-time.After(10 * time.Second):
		t.Fatalf("timeout waiting for %s", operation)
		return ""
	}
}

// countLaneStarts counts the lanes the Scheduler starts: each lane generates
// exactly one claim ID before it claims.
func countLaneStarts(scheduler *Scheduler) *atomic.Int32 {
	var started atomic.Int32
	next := scheduler.options.NewID
	scheduler.options.NewID = func(prefix string) (string, error) {
		if prefix == "claim_" {
			started.Add(1)
		}
		return next(prefix)
	}
	return &started
}

// An idle pool claims once per fill round however many lanes it has
// (SCH-R181, LIM-SCH-21): with one model and two tool lanes, the startup round
// and every poll each make one empty claim of each pool and start no other
// lane. Polls happen only when the test fires them.
func TestSchedulerSupervisorIdlePoolsClaimOncePerRound(t *testing.T) {
	const poll, rounds = time.Hour, 4
	settings := newMutableSchedulerSettings(1, poll)
	settings.setToolLanes(2)
	lanes := newLaneTestStore()
	store := newClaimAttemptStore(lanes)
	scheduler := newLaneTestScheduler(t, lanes, settings, poll)
	scheduler.store = store
	clock := newManualPollClock()
	scheduler.options.Clock = clock
	started := countLaneStarts(scheduler)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- scheduler.Run(ctx) }()

	for round := 1; round <= rounds; round++ {
		if round > 1 {
			// The supervisor's refresh and the terminal release monitor's
			// poll share the poll interval.
			clock.fireWhenArmed(t, poll, 2)
		}
		got := map[resolved.LaneClass]int{}
		for range len(resolved.LaneClasses) {
			got[receiveClass(t, store.attempted, "idle claim")]++
		}
		if got[resolved.LaneClassModel] != 1 || got[resolved.LaneClassTool] != 1 {
			t.Fatalf("round %d claims = %v, want one per pool", round, got)
		}
	}
	cancel()
	if err := receiveError(t, done, 10*time.Second, "Scheduler shutdown"); err != nil {
		t.Fatal(err)
	}
	attempts := store.attemptCounts()
	if attempts[resolved.LaneClassModel] != rounds || attempts[resolved.LaneClassTool] != rounds {
		t.Fatalf("claims over %d rounds = %v, want %d per pool", rounds, attempts, rounds)
	}
	if lanes := started.Load(); lanes != 2*rounds {
		t.Fatalf("lanes started = %d, want %d", lanes, 2*rounds)
	}
}

// A pool with runnable work starts its next lane as soon as the previous lane
// claimed (SCH-R181): one model and two tool Runs hold all three lanes before
// any of them completes. With a third tool lane, that lane's empty claim ends
// the tool pool's round.
func TestSchedulerSupervisorFillsEveryLaneBeforeAnyCompletes(t *testing.T) {
	for _, test := range []struct {
		name      string
		toolLanes int
	}{
		{name: "one lane per Run", toolLanes: 2},
		{name: "a tool lane without work", toolLanes: 3},
	} {
		t.Run(test.name, func(t *testing.T) {
			const poll = time.Hour
			settings := newMutableSchedulerSettings(1, poll)
			settings.setToolLanes(test.toolLanes)
			lanes := newLaneTestStore("run-model-1", "run-tool-1", "run-tool-2")
			lanes.toolRuns("run-tool-1", "run-tool-2")
			store := newClaimAttemptStore(lanes)
			scheduler := newLaneTestScheduler(t, lanes, settings, poll)
			scheduler.store = store
			scheduler.options.Clock = newManualPollClock()
			started := countLaneStarts(scheduler)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			done := make(chan error, 1)
			go func() { done <- scheduler.Run(ctx) }()

			claimed := map[string]bool{}
			for range 3 {
				claimed[receiveString(t, lanes.claimed, 10*time.Second, "claim")] = true
			}
			for range 1 + test.toolLanes {
				receiveClass(t, store.attempted, "claim attempt")
			}
			// Every lane holds its Run while none has completed: the Runs
			// stay blocked until the test ends.
			if active, maximum := lanes.counts(); active != 3 || maximum != 3 || len(claimed) != 3 {
				t.Fatalf("lanes holding Runs = %d (maximum %d), claimed %v, want all three", active, maximum, claimed)
			}
			if releases := lanes.releaseTotal(); releases != 0 {
				t.Fatalf("%d lanes completed before every lane was filled", releases)
			}
			cancel()
			if err := receiveError(t, done, 10*time.Second, "Scheduler shutdown"); err != nil {
				t.Fatal(err)
			}
			attempts := store.attemptCounts()
			if attempts[resolved.LaneClassModel] != 1 || attempts[resolved.LaneClassTool] != test.toolLanes {
				t.Fatalf("claims = %v, want 1 model and %d tool", attempts, test.toolLanes)
			}
			if lanes := started.Load(); int(lanes) != 1+test.toolLanes {
				t.Fatalf("lanes started = %d, want %d", lanes, 1+test.toolLanes)
			}
			if active, _ := lanes.counts(); active != 0 {
				t.Fatalf("shutdown left %d claims held", active)
			}
		})
	}
}
