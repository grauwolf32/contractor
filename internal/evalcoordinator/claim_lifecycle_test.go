package evalcoordinator

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evalstore"
)

// scriptedClaims leases a fixed batch once, reports each claim attempt, and
// records each release with the outcome scripted for its experiment.
type scriptedClaims struct {
	batch      []evalstore.Claim
	passes     chan struct{}
	releaseErr map[string]error

	mu       sync.Mutex
	events   []string
	released map[string]error
	bounded  map[string]bool
}

func (s *scriptedClaims) Claim(context.Context, string, time.Duration, int) ([]evalstore.Claim, error) {
	if s.passes != nil {
		s.passes <- struct{}{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := s.batch
	s.batch = nil
	return out, nil
}

func (s *scriptedClaims) ReleaseClaim(ctx context.Context, claim evalstore.Claim) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, "released "+claim.ExperimentID)
	if s.released == nil {
		s.released, s.bounded = map[string]error{}, map[string]bool{}
	}
	s.released[claim.ExperimentID] = ctx.Err()
	_, s.bounded[claim.ExperimentID] = ctx.Deadline()
	return s.releaseErr[claim.ExperimentID]
}

func (s *scriptedClaims) record(event string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, event)
}

// joiningTicks holds a tick until its pass was cancelled and the test lets
// it end.
type joiningTicks struct {
	store   *scriptedClaims
	entered chan struct{}
	proceed chan struct{}
}

func (s *joiningTicks) Tick(ctx context.Context, claim evalstore.Claim) (bool, error) {
	close(s.entered)
	<-ctx.Done()
	<-s.proceed
	s.store.record("tick ended " + claim.ExperimentID)
	return false, ctx.Err()
}

// A pass cancelled during a tick joins the tick before it releases the
// claim, and the release runs with a live bounded context, so shutdown gives
// the experiment back (PER-R26).
func TestCoordinatorJoinsACancelledTickBeforeReleasingTheClaim(t *testing.T) {
	store := &scriptedClaims{batch: []evalstore.Claim{{ExperimentID: "experiment"}}}
	ticks := &joiningTicks{store: store, entered: make(chan struct{}), proceed: make(chan struct{})}
	// Only the cancellation may end the tick, never its own budget.
	coordinator, err := New(store, ticks, Options{Lease: 5 * time.Minute, OperationTimeout: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() {
		_, err := coordinator.RunOnce(ctx)
		done <- err
	}()
	<-ticks.entered
	cancel()
	close(ticks.proceed)
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled pass error = %v, want the tick's cancellation", err)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if !slices.Equal(store.events, []string{"tick ended experiment", "released experiment"}) {
		t.Fatalf("events = %v, want the tick joined before the release", store.events)
	}
	if store.released["experiment"] != nil || !store.bounded["experiment"] {
		t.Fatalf("release context error = %v, bounded = %t, want a live bounded context",
			store.released["experiment"], store.bounded["experiment"])
	}
}

// meetingTicks lets each tick end only after every claimed experiment's tick
// started, then fails it with its own error.
type meetingTicks struct {
	claims     int
	started    atomic.Int64
	allStarted chan struct{}
}

var errTickFailed = errors.New("tick failed")

func (s *meetingTicks) Tick(ctx context.Context, claim evalstore.Claim) (bool, error) {
	if s.started.Add(1) == int64(s.claims) {
		close(s.allStarted)
	}
	select {
	case <-s.allStarted:
		return true, fmt.Errorf("%w: %s", errTickFailed, claim.ExperimentID)
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

// Every experiment of a pass ticks at the same time, so one slow experiment
// cannot hold back another; each claim is released, a release that finds its
// claim lost is not a failure, and the remaining failures are reported per
// experiment in claim order.
func TestCoordinatorTicksItsClaimsConcurrentlyAndReportsEachFailure(t *testing.T) {
	experiments := []string{"exp-a", "exp-b", "exp-c", "exp-d"}
	errReleaseFailed := errors.New("release failed")
	store := &scriptedClaims{releaseErr: map[string]error{
		"exp-b": evalstore.ErrClaimLost,
		"exp-c": errReleaseFailed,
	}}
	for _, id := range experiments {
		store.batch = append(store.batch, evalstore.Claim{ExperimentID: id})
	}
	ticks := &meetingTicks{claims: len(experiments), allStarted: make(chan struct{})}
	// A sequential pass would wait out this budget instead of meeting the
	// other ticks; the test fails on the error, it measures no time.
	coordinator, err := New(store, ticks, Options{Lease: time.Minute, OperationTimeout: 10 * time.Second, Batch: len(experiments)})
	if err != nil {
		t.Fatal(err)
	}
	changed, err := coordinator.RunOnce(t.Context())
	if !changed || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, evalstore.ErrClaimLost) {
		t.Fatalf("pass = (%t, %v), want concurrent ticks and no lost-claim failure", changed, err)
	}
	store.mu.Lock()
	released := len(store.released)
	store.mu.Unlock()
	if released != len(experiments) {
		t.Fatalf("released %d claims, want %d", released, len(experiments))
	}
	joined, ok := err.(interface{ Unwrap() []error })
	if !ok || len(joined.Unwrap()) != len(experiments) {
		t.Fatalf("pass error = %v, want one failure per experiment", err)
	}
	for index, failure := range joined.Unwrap() {
		id := experiments[index]
		if !errors.Is(failure, errTickFailed) || !strings.Contains(failure.Error(), id) ||
			errors.Is(failure, errReleaseFailed) != (id == "exp-c") {
			t.Fatalf("failure %d = %v, want the tick failure of %s and its release failure only for exp-c",
				index, failure, id)
		}
	}
}

// waitClock records every wait the Run loop asks for and ends it only when
// the test fires it.
type waitClock struct {
	waits chan time.Duration
	fire  chan time.Time
}

func (c *waitClock) After(duration time.Duration) <-chan time.Time {
	c.waits <- duration
	return c.fire
}

type changingTicks struct{}

func (changingTicks) Tick(context.Context, evalstore.Claim) (bool, error) { return true, nil }

// The Eval Run loop keeps its own pacing: even a pass that changed an
// experiment waits for the poll interval or a wake hint before the next one.
func TestCoordinatorRunWaitsForThePollIntervalOrAWakeAfterEveryPass(t *testing.T) {
	store := &scriptedClaims{batch: []evalstore.Claim{{ExperimentID: "active"}}, passes: make(chan struct{})}
	clock := &waitClock{waits: make(chan time.Duration, 1), fire: make(chan time.Time)}
	coordinator, err := New(store, changingTicks{}, Options{PollInterval: 3 * time.Second, Clock: clock})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { done <- coordinator.Run(ctx) }()

	<-store.passes // The first pass ticks a claim that changes its experiment.
	select {
	case wait := <-clock.waits:
		if wait != 3*time.Second {
			t.Fatalf("Run waited %s, want the poll interval", wait)
		}
	case <-store.passes:
		t.Fatal("Run passed again at once after a pass that changed an experiment")
	}
	clock.fire <- time.Time{} // The poll interval ends.
	<-store.passes
	<-clock.waits
	coordinator.Wake() // A wake hint ends the wait without the clock.
	<-store.passes
	<-clock.waits
	cancel()
	if err := <-done; err != nil {
		t.Fatalf("Run after cancellation = %v", err)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.released["active"] != nil {
		t.Fatalf("release of the changed experiment ran with context error %v", store.released["active"])
	}
}
