package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// testRuntimeBatchOptions gives a production batch controller explicit
// budgets; the controller has no defaults of its own.
func testRuntimeBatchOptions() controlplane.RuntimeBatchOptions {
	return controlplane.RuntimeBatchOptions{PrepareTimeout: time.Minute, CleanupTimeout: time.Minute}
}

// slowReleaseRuntime answers a release only after the Scheduler's store
// operation budget has passed, and honours its context deadline the way the
// Runtime HTTP client does.
type slowReleaseRuntime struct {
	deferredLeaseRuntime
	hold         time.Duration
	deadline     time.Time
	hasDeadline  bool
	releaseCalls int
}

func (r *slowReleaseRuntime) Release(ctx context.Context, _ controlplane.Reservation) error {
	r.releaseCalls++
	r.deadline, r.hasDeadline = ctx.Deadline()
	held, cancel := context.WithTimeout(context.Background(), r.hold)
	defer cancel()
	<-held.Done()
	if r.hasDeadline && !time.Now().Before(r.deadline) {
		return context.DeadlineExceeded
	}
	return ctx.Err()
}

// A release batch keeps its whole runtimeLifecycle.cleanupTimeout budget even
// when scheduler.operationTimeout is shorter (SCH-R65, OPS-R15), on the lane
// path after a terminal commit and on terminal release recovery.
func TestTerminalReleaseUsesCleanupBudgetNotOperationTimeout(t *testing.T) {
	const operationTimeout, cleanupTimeout = time.Millisecond, time.Hour
	for _, recovery := range []bool{false, true} {
		t.Run(map[bool]string{false: "lane", true: "recovery"}[recovery], func(t *testing.T) {
			h := newSchedulerHarness(t)
			h.scheduler.options.OperationTimeout = operationTimeout
			execution := h.persistedExecution(t, runstore.StageSucceeded)
			h.store.stages = []runstore.StageExecution{execution}
			h.store.run.State = runstore.RunSucceeded
			now := h.clock.now
			registry, reservation := registerTerminalReleaseGrant(t, h, execution, controlplane.RegistryOptions{
				Now:          func() time.Time { return now },
				MonotonicNow: func() time.Duration { return 0 },
			})
			runtime := &slowReleaseRuntime{hold: 2 * operationTimeout}
			batch, err := controlplane.NewRuntimeBatchController(runtime, registry, controlplane.RuntimeBatchOptions{
				PrepareTimeout: time.Minute, CleanupTimeout: cleanupTimeout,
			})
			if err != nil {
				t.Fatal(err)
			}
			h.scheduler.allocator = lostRuntimeReleaseAllocator{Allocator: h.allocator, registry: registry}
			h.scheduler.workers = batch

			started := time.Now()
			if recovery {
				var worked bool
				worked, err = h.scheduler.recoverTerminalRelease(context.Background())
				if !worked {
					t.Fatal("terminal release recovery found no work")
				}
			} else {
				err = h.scheduler.releaseTerminal(execution.StageExecutionID, []controlplane.Reservation{reservation})
			}
			if err != nil {
				t.Fatalf("release outliving the operation budget failed: %v", err)
			}
			if runtime.releaseCalls != 1 || !runtime.hasDeadline || runtime.deadline.Before(started.Add(cleanupTimeout)) {
				t.Fatalf("Runtime release calls = %d, deadline = %v (set %t); want one call with the %v cleanup budget from %v",
					runtime.releaseCalls, runtime.deadline, runtime.hasDeadline, cleanupTimeout, started)
			}
			if _, err := registry.GetGrant(reservation.Grant.AllocationID); !errors.Is(err, controlplane.ErrAllocationNotFound) {
				t.Fatalf("released grant = %v", err)
			}
			if h.store.allocations[0].ReleaseCompletedAt == nil {
				t.Fatalf("durable release marker = %+v", h.store.allocations[0])
			}
		})
	}
}

// The ServerConfig owns every process budget, so the Scheduler has no package
// default that could silently differ from it.
func TestNewRequiresEveryProcessBudget(t *testing.T) {
	h := newSchedulerHarness(t)
	for name, unset := range map[string]func(*Options){
		"pollInterval":        func(options *Options) { options.PollInterval = 0 },
		"claimDuration":       func(options *Options) { options.ClaimDuration = 0 },
		"claimMicrosecond":    func(options *Options) { options.ClaimDuration = time.Nanosecond },
		"leaseScanInterval":   func(options *Options) { options.LeaseScanInterval = 0 },
		"operationTimeout":    func(options *Options) { options.OperationTimeout = 0 },
		"plannerTimeout":      func(options *Options) { options.PlannerTimeout = 0 },
		"finalizationTimeout": func(options *Options) { options.FinalizationTimeout = 0 },
		"abortTimeout":        func(options *Options) { options.AbortTimeout = 0 },
	} {
		t.Run(name, func(t *testing.T) {
			options := h.scheduler.options
			unset(&options)
			if _, err := New(h.store, h.persistence, h.artifacts, h.allocator, h.workers, h.planners, options); err == nil {
				t.Fatal("Scheduler accepted a missing process budget")
			}
		})
	}
}

// batchContextWorkers records the deadlines of the contexts the Scheduler
// hands to Runtime batches.
type batchContextWorkers struct {
	*memoryWorkers
	prepares, releases []observedDeadline
}

type observedDeadline struct {
	at  time.Time
	set bool
}

func (w *batchContextWorkers) PrepareAll(
	ctx context.Context,
	reservations []controlplane.Reservation,
	settings map[string]runtimesettings.WorkerExecutionSettings,
) (map[string]contracts.WorkerHandle, error) {
	at, set := ctx.Deadline()
	w.prepares = append(w.prepares, observedDeadline{at: at, set: set})
	return w.memoryWorkers.PrepareAll(ctx, reservations, settings)
}

func (w *batchContextWorkers) ReleaseAll(ctx context.Context, reservations []controlplane.Reservation) error {
	at, set := ctx.Deadline()
	w.releases = append(w.releases, observedDeadline{at: at, set: set})
	return w.memoryWorkers.ReleaseAll(ctx, reservations)
}

// The Scheduler hands a Runtime batch only its cancellation: the batch
// budgets belong to the WorkerController, so neither prepare nor release may
// inherit the store-operation deadline.
func TestSchedulerLeavesRuntimeBatchBudgetsToWorkerController(t *testing.T) {
	h := newSchedulerHarness(t)
	workers := &batchContextWorkers{memoryWorkers: h.workers}
	h.scheduler.workers = workers
	if worked, err := h.scheduler.RunOnce(context.Background()); err != nil || !worked {
		t.Fatalf("run: %v %v", worked, err)
	}
	if len(workers.prepares) != 1 || workers.prepares[0].set {
		t.Fatalf("prepare batches = %+v; want one without an inherited deadline", workers.prepares)
	}
	if len(workers.releases) != 1 || workers.releases[0].set {
		t.Fatalf("release batches = %+v; want one without an inherited deadline", workers.releases)
	}
}
