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
		{name: "transient past lease margin", failures: []error{transient, transient, transient, transient}, lost: transient},
		{name: "conflict is immediate loss", failures: []error{&runstore.StateConflictError{Resource: "WorkflowRun claim", ID: "run-1"}}, lost: runstore.ErrConflict},
	} {
		t.Run(test.name, func(t *testing.T) {
			store := &flakyRenewalStore{laneTestStore: newLaneTestStore("run-1"), failures: test.failures, renewed: make(chan int, 64)}
			// Ten-minute ticks against a one-hour lease: the safety margin
			// (a third of the lease) tolerates three failed ticks, not four.
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
				s.renewClaim(ownership, cancelOwnership, func(error) {}, nil, "run-1", "claim-1", leaseExpiresAt)
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

func (s *flakyRenewalStore) RenewRunClaim(context.Context, string, string, time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.count++
	if s.count <= len(s.failures) {
		return s.failures[s.count-1]
	}
	select {
	case s.renewed <- s.count:
	default:
	}
	return nil
}

func (s *flakyRenewalStore) calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.count
}

func (s *flakyRenewalStore) GetRun(context.Context, string) (runstore.WorkflowRun, error) {
	return runstore.WorkflowRun{RunID: "run-1", State: runstore.RunRunning}, nil
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
