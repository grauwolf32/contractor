package scheduler

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// TestSchedulerReleasesPlacementWhenAdmissionRacesPause proves that when an
// owner's queue pauses between the pre-placement probe and atomic admission,
// the Stage's durably pinned placement is discarded rather than held for the
// whole pause, and the Stage is placed again on resume without consuming a
// retry attempt. The fake placement pins rows like production placement.
func TestSchedulerReleasesPlacementWhenAdmissionRacesPause(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.store.run.State = runstore.RunPending
	harness.store.stages = []runstore.StageExecution{harness.persistedExecution(t, runstore.StagePreparing)}

	// The queue is running when the pre-placement probe reads it, then pauses
	// after placement pinned its rows but before AdmitStage commits.
	pin := harness.allocator.pin
	paused := false
	harness.allocator.pin = func(ctx context.Context, reservations []controlplane.Reservation) error {
		if err := pin(ctx, reservations); err != nil {
			return err
		}
		if !paused {
			harness.persistence.queuePaused = true
			paused = true
		}
		return nil
	}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if !worked || !errors.Is(err, ErrDeferred) {
		t.Fatalf("raced RunOnce = (%v, %v), want deferred", worked, err)
	}
	// The pinned rows and the live reservation are both gone, so the Runtime
	// is eligible for other owners while this owner stays paused.
	stage := harness.store.stages[0]
	if harness.allocator.reserveCalls != 1 || len(harness.allocator.grants) != 0 ||
		len(harness.allocator.cached) != 0 || len(harness.store.allocations) != 0 ||
		harness.allocator.released[stage.StageExecutionID] || stage.AdmittedAt != nil ||
		stage.State != runstore.StagePreparing || stage.Termination != nil ||
		harness.store.run.State != runstore.RunPending {
		t.Fatalf("raced placement not discarded = reserves:%d grants:%d cached:%d allocations:%d tombstone:%v admitted:%v stage:%s run:%s",
			harness.allocator.reserveCalls, len(harness.allocator.grants), len(harness.allocator.cached),
			len(harness.store.allocations), harness.allocator.released[stage.StageExecutionID],
			stage.AdmittedAt, stage.State, harness.store.run.State)
	}

	// On resume the same Stage is placed again and runs to success; no
	// retry attempt is created.
	harness.persistence.queuePaused = false
	worked, err = harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked || harness.store.run.State != runstore.RunSucceeded ||
		harness.allocator.reserveCalls != 2 || len(harness.store.stages) != 1 ||
		harness.store.stages[0].State != runstore.StageSucceeded || harness.store.stages[0].Attempt != 1 {
		t.Fatalf("resumed RunOnce = (%v, %v), run=%s reserves=%d stages=%d",
			worked, err, harness.store.run.State, harness.allocator.reserveCalls, len(harness.store.stages))
	}
	if len(harness.store.allocations) != 1 ||
		harness.store.allocations[0].StageExecutionID != stage.StageExecutionID {
		t.Fatalf("re-placed Stage allocations = %+v", harness.store.allocations)
	}
}
