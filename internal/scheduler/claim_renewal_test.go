package scheduler

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestClaimRenewalRetriesTransientErrorsUntilLeaseMargin(t *testing.T) {
	transient := errors.New("connection reset by peer")
	for _, test := range []struct {
		name     string
		failures []error
		lost     error
	}{
		{name: "recovers before lease margin", failures: []error{transient, transient}},
		{name: "transient past lease margin", failures: []error{transient, transient, transient}, lost: transient},
		{name: "conflict is immediate loss", failures: []error{&runstore.StateConflictError{Resource: "WorkflowRun claim", ID: "run-1"}}, lost: runstore.ErrConflict},
	} {
		t.Run(test.name, func(t *testing.T) {
			store := &flakyRenewalStore{laneTestStore: newLaneTestStore("run-1"), failures: test.failures, renewed: make(chan int, 64)}
			// A one-hour lease renewed every twenty minutes and, after a
			// failure, retried every ten-minute poll interval: the safety
			// margin (a third of the lease) tolerates two failed attempts,
			// at twenty and thirty minutes, not a third at forty.
			s := newLaneTestScheduler(t, store.laneTestStore, newMutableSchedulerSettings(1, time.Second), 10*time.Minute)
			s.store = store
			clock := &steppingClock{now: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
			s.options.Clock = clock
			ownership, cancelOwnership := context.WithCancelCause(t.Context())
			defer cancelOwnership(nil)
			leaseExpiresAt := clock.Now().Add(s.options.ClaimDuration)
			done := make(chan struct{})
			go func() {
				defer close(done)
				s.renewClaim(ownership, cancelOwnership, func(error) {}, nil, "run-1", "claim-1", runstore.RunRunning, leaseExpiresAt)
			}()
			if test.lost == nil {
				// Keep renewing well past the original lease expiry.
				for calls := 0; calls < 10; {
					select {
					case calls = <-store.renewed:
					case <-done:
						t.Fatalf("renewal stopped after transient errors: %v", context.Cause(ownership))
					case <-time.After(time.Second):
						t.Fatal("renewal did not continue")
					}
				}
				cancelOwnership(nil)
			}
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("renewal did not stop")
			}
			cause := context.Cause(ownership)
			if test.lost == nil {
				if errors.Is(cause, ErrClaimLost) {
					t.Fatalf("transient renewal errors cancelled ownership: %v", cause)
				}
				return
			}
			if !errors.Is(cause, ErrClaimLost) || !errors.Is(cause, test.lost) {
				t.Fatalf("ownership cause = %v, want claim loss from %v", cause, test.lost)
			}
			if calls := store.calls(); calls != len(test.failures) {
				t.Fatalf("renewal attempts = %d, want %d", calls, len(test.failures))
			}
		})
	}
}

type flakyRenewalStore struct {
	*laneTestStore
	mu       sync.Mutex
	failures []error
	count    int
	renewed  chan int
}

func (s *flakyRenewalStore) RenewRunClaim(context.Context, string, string, time.Duration) (runstore.WorkflowRunState, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.count++
	if s.count <= len(s.failures) {
		return "", s.failures[s.count-1]
	}
	select {
	case s.renewed <- s.count:
	default:
	}
	return runstore.RunRunning, nil
}

func (s *flakyRenewalStore) calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.count
}

// GetRunState answers the renewal's conflict check: the Run is still running,
// so a conflict is a lost claim, not a terminal transition.
func (s *flakyRenewalStore) GetRunState(context.Context, string) (runstore.WorkflowRunState, error) {
	return runstore.RunRunning, nil
}

// TestClaimRenewalRenewsEveryThirdOfTheLease runs a lane's claim renewal for
// ten virtual minutes with the default ServerConfig budgets, a two-minute
// lease and a one-second poll interval: the lane renews every 40 seconds,
// never more than three times in a minute (LIM-SCH-15), where renewing every
// poll interval made sixty.
func TestClaimRenewalRenewsEveryThirdOfTheLease(t *testing.T) {
	store := &countingRenewalStore{laneTestStore: newLaneTestStore("run-1")}
	s := newLaneTestScheduler(t, store.laneTestStore, newMutableSchedulerSettings(1, time.Second), time.Second)
	s.store = store
	s.options.ClaimDuration = 2 * time.Minute
	clock := &steppingClock{now: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
	s.options.Clock = clock
	start := clock.Now()
	ownership, cancelOwnership := context.WithCancelCause(t.Context())
	defer cancelOwnership(nil)
	store.clock, store.until, store.stop = clock, start.Add(10*time.Minute), func() { cancelOwnership(nil) }

	s.renewClaim(ownership, cancelOwnership, func(error) {}, nil, "run-1", "claim-1", runstore.RunRunning,
		start.Add(s.options.ClaimDuration))
	if cause := context.Cause(ownership); errors.Is(cause, ErrClaimLost) {
		t.Fatalf("renewal lost the claim: %v", cause)
	}
	if len(store.renewals) != 14 || store.renewals[0].Sub(start) != 40*time.Second {
		first := store.renewals[:min(len(store.renewals), 3)]
		t.Fatalf("%d renewals in ten minutes, the first at %v, want one every 40 seconds", len(store.renewals), first)
	}
	for index, at := range store.renewals {
		inMinute := 0
		for _, later := range store.renewals[index:] {
			if later.Sub(at) < time.Minute {
				inMinute++
			}
		}
		if inMinute > 3 {
			t.Fatalf("%d renewals in the minute after %s, want at most 3", inMinute, at.Sub(start))
		}
	}
}

// countingRenewalStore records the virtual time of every renewal and ends
// the lane's ownership once the clock reaches until.
type countingRenewalStore struct {
	*laneTestStore
	clock    *steppingClock
	until    time.Time
	stop     func()
	renewals []time.Time
}

func (s *countingRenewalStore) RenewRunClaim(ctx context.Context, _, _ string, _ time.Duration) (runstore.WorkflowRunState, error) {
	if !s.clock.now.Before(s.until) {
		s.stop()
		return "", ctx.Err()
	}
	s.renewals = append(s.renewals, s.clock.now)
	return runstore.RunRunning, nil
}

// steppingClock advances virtual time by each requested wait and fires at once.
type steppingClock struct {
	now time.Time
}

func (c *steppingClock) Now() time.Time { return c.now }

func (c *steppingClock) After(duration time.Duration) <-chan time.Time {
	c.now = c.now.Add(duration)
	fired := make(chan time.Time, 1)
	fired <- c.now
	return fired
}
