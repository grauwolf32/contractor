package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestClaimCleanupPreservesOwnershipAfterExecutionCancellation(t *testing.T) {
	for _, cause := range []error{ErrRunCancellationRequested, ErrAllocationLeaseLost} {
		t.Run(cause.Error(), func(t *testing.T) {
			ownership, cancelOwnership := context.WithCancelCause(t.Context())
			defer cancelOwnership(nil)
			execution, cancelExecution := context.WithCancelCause(
				context.WithValue(ownership, runOwnershipContextKey{}, ownership),
			)
			cancelExecution(cause)
			cleanup, cancelCleanup := claimCleanupContext(execution)
			defer cancelCleanup()
			if cleanup.Err() != nil {
				t.Fatal("ordinary cancellation prevented cleanup", cleanup.Err())
			}
			cancelOwnership(errors.Join(ErrClaimLost, errors.New("claim replaced")))
			select {
			case <-cleanup.Done():
			case <-time.After(time.Second):
				t.Fatal("late claim loss did not stop cleanup")
			}
			if !errors.Is(context.Cause(cleanup), ErrClaimLost) || context.Cause(execution) != cause {
				t.Fatalf("cleanup cause = %v, execution cause = %v", context.Cause(cleanup), context.Cause(execution))
			}
			// Operations started after observed claim loss must be cancelled
			// synchronously, before the AfterFunc callback gets scheduled.
			next, cancelNext := claimCleanupContext(execution)
			defer cancelNext()
			if !errors.Is(context.Cause(next), ErrClaimLost) {
				t.Fatal("new cleanup operation ignored an already lost claim")
			}
		})
	}
}

func TestClaimCleanupRetainsProcessShutdownAndFiniteOperationBudget(t *testing.T) {
	ownership, cancelOwnership := context.WithCancelCause(t.Context())
	ctx := context.WithValue(ownership, runOwnershipContextKey{}, ownership)
	cancelOwnership(context.Canceled)
	s := &Scheduler{options: Options{OperationTimeout: 10 * time.Millisecond}}
	cleanup, cancelCleanup := s.terminalOperationContext(ctx)
	defer cancelCleanup()
	if cleanup.Err() != nil {
		t.Fatal("process cancellation prevented bounded cleanup", cleanup.Err())
	}
	select {
	case <-cleanup.Done():
		if !errors.Is(cleanup.Err(), context.DeadlineExceeded) {
			t.Fatal("cleanup lost its independent deadline", cleanup.Err())
		}
	case <-time.After(time.Second):
		t.Fatal("cleanup outlived its operation budget")
	}
}

func TestCompletedRunCancelsInflightRenewalWithoutClaimLoss(t *testing.T) {
	store := &blockedRenewalStore{laneTestStore: newLaneTestStore("run-1"), started: make(chan struct{})}
	s := newLaneTestScheduler(t, store.laneTestStore, newMutableSchedulerSettings(1, time.Second), 10*time.Millisecond)
	s.store = store
	s.options.OperationTimeout = 5 * time.Second
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() {
		defer close(done)
		_, err := s.progressOneClaim(ctx)
		done <- err
	}()
	defer func() {
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Error("lane did not stop")
		}
	}()
	select {
	case <-store.started:
	case <-time.After(time.Second):
		t.Fatal("renewal was not reached")
	}
	store.unblock("run-1")
	if err := receiveError(t, done, time.Second, "completed Run with in-flight renewal"); err != nil {
		t.Fatal("normal renewal stop became claim loss", err)
	}
}

type blockedRenewalStore struct {
	*laneTestStore
	started chan struct{}
}

func (s *blockedRenewalStore) RenewRunClaim(ctx context.Context, _, _ string, _ time.Duration) error {
	close(s.started)
	<-ctx.Done()
	return ctx.Err()
}
