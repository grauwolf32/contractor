package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestClaimedCancellingCleanupSurvivesRenewal(t *testing.T) {
	for _, test := range []struct {
		name  string
		state runstore.StageExecutionState
	}{
		{name: "abort", state: runstore.StageRunning},
		{name: "finalize", state: runstore.StageFinalizing},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newSchedulerHarness(t)
			now := time.Now().UTC()
			h.clock = staticClock{now: now}
			h.allocator.clock = h.clock
			h.workers.clock = h.clock
			h.requestCancellation("already cancelling when claimed")
			execution := h.persistedExecution(t, test.state)
			if test.state == runstore.StageFinalizing {
				candidate := h.planners.result.Clone()
				finalizationID := "finalization-before-claim"
				deadline := now.Add(3 * time.Second)
				execution.CandidateResultSchemaVersion = stringPointer(contracts.APIVersion)
				execution.CandidateResult = &candidate
				execution.FinalizationID = &finalizationID
				execution.FinalizationDeadline = &deadline
			}
			h.store.stages = []runstore.StageExecution{execution}
			h.persistence.stageStates = []runstore.StageExecutionState{test.state}
			h.installRecordedReservation(execution.StageExecutionID)

			clock := &manualRenewalClock{now: now, waits: make(chan chan time.Time, 2)}
			h.scheduler.options.Clock = clock
			h.scheduler.options.PollInterval = 10 * time.Millisecond
			h.scheduler.options.ClaimDuration = 300 * time.Millisecond
			h.scheduler.options.AbortTimeout = 3 * time.Second
			workers := &waitingTerminalWorkers{
				memoryWorkers: h.workers,
				started:       make(chan context.Context, 1),
				release:       make(chan struct{}),
			}
			h.scheduler.workers = workers
			ctx, cancel := context.WithTimeout(t.Context(), 4*time.Second)
			defer cancel()
			done := make(chan error, 1)
			go func() {
				worked, err := h.scheduler.RunOnce(ctx)
				if !worked && err == nil {
					err = errors.New("cancelling Run was not claimed")
				}
				done <- err
			}()

			var workerContext context.Context
			select {
			case workerContext = <-workers.started:
			case <-ctx.Done():
				t.Fatal("terminal Worker was not called")
			}
			var tick chan time.Time
			select {
			case tick = <-clock.waits:
			case <-ctx.Done():
				t.Fatal("claim renewal did not start")
			}
			tick <- now.Add(h.scheduler.options.PollInterval)
			// The second wait proves renewal has observed the cancelling Run.
			select {
			case <-clock.waits:
			case <-ctx.Done():
				t.Fatal("claim renewal did not complete")
			}
			if err := workerContext.Err(); err != nil {
				t.Fatalf("claim renewal cut short %s: %v", test.name, err)
			}
			close(workers.release)
			select {
			case err := <-done:
				if err != nil {
					t.Fatalf("cancellation cleanup: %v", err)
				}
			case <-ctx.Done():
				t.Fatal("cancellation cleanup did not finish")
			}
			if h.store.run.State != runstore.RunCancelled || len(h.store.reports) != 1 ||
				!h.store.reports[0].Report.Worker.Complete || !h.store.reports[0].Report.Runtime.Complete {
				t.Fatalf("terminal state/report = %s, %+v", h.store.run.State, h.store.reports)
			}
		})
	}
}

func TestClaimedCancellingCleanupStopsOnClaimLoss(t *testing.T) {
	h := newSchedulerHarness(t)
	now := time.Now().UTC()
	h.clock = staticClock{now: now}
	h.allocator.clock = h.clock
	h.workers.clock = h.clock
	h.requestCancellation("already cancelling when claimed")
	execution := h.persistedExecution(t, runstore.StageRunning)
	h.store.stages = []runstore.StageExecution{execution}
	h.persistence.stageStates = []runstore.StageExecutionState{runstore.StageRunning}
	h.installRecordedReservation(execution.StageExecutionID)
	clock := &manualRenewalClock{now: now, waits: make(chan chan time.Time, 1)}
	h.scheduler.options.Clock = clock
	h.scheduler.options.PollInterval = 10 * time.Millisecond
	h.scheduler.options.ClaimDuration = 300 * time.Millisecond
	h.scheduler.options.AbortTimeout = 3 * time.Second
	h.scheduler.store = &lostRenewalStore{memorySchedulerStore: h.store}
	workers := &waitingTerminalWorkers{
		memoryWorkers: h.workers,
		started:       make(chan context.Context, 1),
		release:       make(chan struct{}),
	}
	h.scheduler.workers = workers
	ctx, cancel := context.WithTimeout(t.Context(), 4*time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := h.scheduler.RunOnce(ctx)
		done <- err
	}()
	var workerContext context.Context
	select {
	case workerContext = <-workers.started:
	case <-ctx.Done():
		t.Fatal("abort Worker was not called")
	}
	select {
	case tick := <-clock.waits:
		tick <- now.Add(10 * time.Millisecond)
	case <-ctx.Done():
		t.Fatal("claim renewal did not start")
	}
	select {
	case <-workerContext.Done():
		if !errors.Is(context.Cause(workerContext), ErrClaimLost) {
			t.Fatalf("Worker stopped for %v, want claim loss", context.Cause(workerContext))
		}
	case <-ctx.Done():
		t.Fatal("claim loss did not stop abort cleanup")
	}
	select {
	case err := <-done:
		if !errors.Is(err, ErrClaimLost) {
			t.Fatalf("claim loss result = %v", err)
		}
	case <-ctx.Done():
		t.Fatal("claim-loss lane did not stop")
	}
}

func TestClaimRenewalInterruptsCancellationAfterRunningClaim(t *testing.T) {
	h := newSchedulerHarness(t)
	h.store.run.State = runstore.RunCancelling
	h.store.claimID = "claim-before-cancellation"
	now := time.Now().UTC()
	clock := &manualRenewalClock{now: now, waits: make(chan chan time.Time, 2)}
	h.scheduler.options.Clock = clock
	h.scheduler.options.PollInterval = 10 * time.Millisecond
	ownership, cancelOwnership := context.WithCancelCause(t.Context())
	defer cancelOwnership(nil)
	execution, cancelExecution := context.WithCancelCause(ownership)
	done := make(chan struct{})
	go func() {
		defer close(done)
		h.scheduler.renewClaim(
			ownership, cancelOwnership, cancelExecution, nil,
			h.store.run.RunID, h.store.claimID, runstore.RunRunning, now.Add(time.Hour),
		)
	}()
	select {
	case tick := <-clock.waits:
		tick <- now.Add(10 * time.Millisecond)
	case <-time.After(time.Second):
		t.Fatal("claim renewal did not start")
	}
	select {
	case <-clock.waits:
	case <-time.After(time.Second):
		t.Fatal("claim renewal did not observe cancellation")
	}
	if !errors.Is(context.Cause(execution), ErrRunCancellationRequested) {
		t.Fatalf("execution cause = %v", context.Cause(execution))
	}
	cancelOwnership(nil)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("claim renewal did not stop")
	}
}

type manualRenewalClock struct {
	now   time.Time
	waits chan chan time.Time
}

func (c *manualRenewalClock) Now() time.Time { return c.now }

func (c *manualRenewalClock) After(time.Duration) <-chan time.Time {
	tick := make(chan time.Time, 1)
	c.waits <- tick
	return tick
}

type waitingTerminalWorkers struct {
	*memoryWorkers
	started chan context.Context
	release chan struct{}
}

func (w *waitingTerminalWorkers) FinalizeAll(
	ctx context.Context, reservations []controlplane.Reservation, _ string, _ time.Time,
) (map[string]contracts.AllocationFinalReport, error) {
	return w.complete(ctx, reservations)
}

func (w *waitingTerminalWorkers) AbortAll(
	ctx context.Context, reservations []controlplane.Reservation, _ string,
	_ contracts.TerminationError, _ time.Time,
) (map[string]contracts.AllocationFinalReport, error) {
	return w.complete(ctx, reservations)
}

func (w *waitingTerminalWorkers) complete(
	ctx context.Context, reservations []controlplane.Reservation,
) (map[string]contracts.AllocationFinalReport, error) {
	w.started <- ctx
	select {
	case <-w.release:
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	reports := make(map[string]contracts.AllocationFinalReport, len(reservations))
	for _, reservation := range reservations {
		reports[reservation.Grant.LogicalAgentName] = schedulerTestAllocationReport(
			reservation.Grant.AllocationID, time.Now().UTC(),
		)
	}
	return reports, nil
}

type lostRenewalStore struct{ *memorySchedulerStore }

func (*lostRenewalStore) RenewRunClaim(context.Context, string, string, time.Duration) error {
	return runstore.ErrConflict
}
