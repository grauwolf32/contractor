package scheduler

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type terminalDeadlineWorkers struct {
	*memoryWorkers
	deadline    time.Time
	operationID string
}

func (w *terminalDeadlineWorkers) FinalizeAll(ctx context.Context, rs []controlplane.Reservation, id string, deadline time.Time) (map[string]contracts.AllocationFinalReport, error) {
	w.deadline, w.operationID = deadline, id
	return w.memoryWorkers.FinalizeAll(ctx, rs, id, deadline)
}

func (w *terminalDeadlineWorkers) AbortAll(ctx context.Context, rs []controlplane.Reservation, id string, reason contracts.TerminationError, deadline time.Time) (map[string]contracts.AllocationFinalReport, error) {
	w.deadline, w.operationID = deadline, id
	return w.memoryWorkers.AbortAll(ctx, rs, id, reason, deadline)
}

func TestConfiguredTerminalBudgetsAreSavedAndRecoveryDoesNotRenew(t *testing.T) {
	for _, abort := range []bool{false, true} {
		t.Run(map[bool]string{false: "finalize", true: "abort"}[abort], func(t *testing.T) {
			h := newSchedulerHarness(t)
			workers := &terminalDeadlineWorkers{memoryWorkers: h.workers}
			h.scheduler.workers = workers
			h.scheduler.options.FinalizationTimeout = 17 * time.Second
			h.scheduler.options.AbortTimeout = 23 * time.Second
			budget := 17 * time.Second
			if abort {
				budget = 23 * time.Second
				h.planners.runErrors = []error{planner.NewError("planner_model_call_limit", "test limit", true, nil)}
			}
			if worked, err := h.scheduler.RunOnce(context.Background()); err != nil || !worked {
				t.Fatalf("run: %v %v", worked, err)
			}
			saved := workers.deadline
			id := workers.operationID
			if !saved.Equal(h.clock.now.Add(budget)) || id == "" {
				t.Fatalf("terminal = %v, %q", saved, id)
			}
			execution := h.store.stages[0]
			// Replay the persisted terminal phase under substantially different process
			// settings. The original absolute deadline and operation ID remain binding.
			if abort {
				execution.State = runstore.StageAborting
			} else {
				execution.State = runstore.StageFinalizing
			}
			h.store.stages = []runstore.StageExecution{execution}
			h.store.run.State = runstore.RunRunning
			h.persistence.stageStates = []runstore.StageExecutionState{execution.State}
			h.installRecordedReservation(execution.StageExecutionID)
			options := h.scheduler.options
			options.FinalizationTimeout, options.AbortTimeout = time.Minute, 2*time.Minute
			restarted, err := New(h.store, h.persistence, h.artifacts, h.allocator, workers, h.planners, options)
			if err != nil {
				t.Fatal(err)
			}
			if worked, err := restarted.RunOnce(context.Background()); err != nil || !worked {
				t.Fatalf("recovery: %v %v", worked, err)
			}
			if !workers.deadline.Equal(saved) || workers.operationID != id {
				t.Fatalf("recovery changed terminal deadline/ID: %v %q", workers.deadline, workers.operationID)
			}
		})
	}
}
