package scheduler

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// TestSchedulerAllocationLossInterruptsOnlyOwningStage proves an allocation
// loss cancels a lane only when the loss names the Stage that lane is
// currently progressing. A loss for an already-finished Stage (owner died
// after completion but before release) must leave the current Stage running.
func TestSchedulerAllocationLossInterruptsOnlyOwningStage(t *testing.T) {
	h := newSchedulerHarness(t)
	runID := "run-loss-scope"
	var cause error
	h.scheduler.registerActiveRun(runID, "claim-scope", func(c error) { cause = c })
	h.scheduler.setActiveStage(runID, "claim-scope", "stage-current")

	// Loss of a finished Stage's allocation must not interrupt the current one.
	h.allocator.losses = []controlplane.AllocationLoss{{
		RunID: runID, StageExecutionID: "stage-finished", AllocationID: "alloc-finished",
		Reason: controlplane.LossControlLeaseExpired,
	}}
	h.scheduler.pollAllocationLosses()
	if cause != nil {
		t.Fatalf("loss of a finished Stage interrupted the current Stage: %v", cause)
	}

	// Loss of the current Stage's allocation interrupts it and carries the reason.
	h.allocator.losses = []controlplane.AllocationLoss{{
		RunID: runID, StageExecutionID: "stage-current", AllocationID: "alloc-current",
		Reason: controlplane.LossRuntimeMismatch,
	}}
	h.scheduler.pollAllocationLosses()
	var lossErr *AllocationLeaseLossError
	if !errors.As(cause, &lossErr) {
		t.Fatalf("loss of the current Stage did not interrupt it: %v", cause)
	}
	if lossErr.Loss.StageExecutionID != "stage-current" || lossErr.Loss.Reason != controlplane.LossRuntimeMismatch {
		t.Fatalf("interrupt carried the wrong loss: %+v", lossErr.Loss)
	}
}

// TestSchedulerAllocationLossIgnoresUnknownRun proves a loss for a Run no lane
// is currently progressing interrupts nothing; its fenced grant is left to
// terminal release recovery.
func TestSchedulerAllocationLossIgnoresUnknownRun(t *testing.T) {
	h := newSchedulerHarness(t)
	h.allocator.losses = []controlplane.AllocationLoss{{
		RunID: "run-not-active", StageExecutionID: "stage-x", AllocationID: "alloc-x",
		Reason: controlplane.LossControlLeaseExpired,
	}}
	// No active lane registered: pollAllocationLosses must not panic and must
	// interrupt nothing.
	h.scheduler.pollAllocationLosses()
}

// TestSchedulerLeaseLossTerminationCarriesReason proves the Stage termination
// diagnostic reports the specific loss reason rather than collapsing every
// loss into control_lease_expired.
func TestSchedulerLeaseLossTerminationCarriesReason(t *testing.T) {
	for _, test := range []struct {
		name     string
		reason   controlplane.AllocationLossReason
		wantCode string
	}{
		{name: "control lease expired", reason: controlplane.LossControlLeaseExpired, wantCode: "control_lease_expired"},
		{name: "runtime state mismatch", reason: controlplane.LossRuntimeMismatch, wantCode: "runtime_state_mismatch"},
	} {
		t.Run(test.name, func(t *testing.T) {
			reason := test.reason
			harness := newSchedulerHarness(t)
			harness.planners.onRun = func() {
				harness.persistence.stageStates = append(harness.persistence.stageStates, runstore.StageRunning)
				reservation := harness.allocator.cached[0]
				reservation.Grant.Lost = true
				reservation.Grant.WriteFenced = true
				reservation.Grant.LossReason = reason
				harness.allocator.cached[0] = reservation
				grant := harness.allocator.grants[reservation.Grant.AllocationID]
				grant.Lost = true
				grant.WriteFenced = true
				grant.LossReason = reason
				harness.allocator.grants[reservation.Grant.AllocationID] = grant
				harness.allocator.losses = append(harness.allocator.losses, controlplane.AllocationLoss{
					AllocationID:      reservation.Grant.AllocationID,
					RuntimeInstanceID: reservation.Grant.RuntimeInstanceID,
					RunID:             harness.store.run.RunID,
					StageExecutionID:  reservation.Grant.StageExecutionID,
					Reason:            reason,
				})
				harness.scheduler.pollAllocationLosses()
			}

			worked, err := harness.scheduler.RunOnce(context.Background())
			if err != nil || !worked {
				t.Fatalf("RunOnce = (%v, %v)", worked, err)
			}
			execution := harness.store.stages[0]
			if harness.store.run.State != runstore.RunFailed || execution.State != runstore.StageInterrupted ||
				execution.Termination == nil || execution.Termination.Code != test.wantCode ||
				!execution.Termination.Retryable {
				t.Fatalf("lease-loss termination = run:%s stage:%+v", harness.store.run.State, execution)
			}
		})
	}
}
