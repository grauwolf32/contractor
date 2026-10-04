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
// the Stage's just-reserved Runtime slot is discarded rather than pinned for
// the whole pause, and the Stage is placed again on resume.
func TestSchedulerReleasesPlacementWhenAdmissionRacesPause(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.store.run.State = runstore.RunPending
	harness.store.stages = []runstore.StageExecution{harness.persistedExecution(t, runstore.StagePreparing)}

	// The queue is running when the pre-placement probe reads it, then pauses
	// after a Runtime slot has been reserved but before AdmitStage commits.
	paused := false
	harness.allocator.record = func(_ context.Context, _ string, _ []controlplane.Reservation) error {
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
	// A1: the reservation happened but the pinned batch was released, so the
	// Runtime is eligible for other owners while this owner stays paused.
	if harness.allocator.reserveCalls != 1 || len(harness.allocator.grants) != 0 ||
		len(harness.allocator.cached) != 0 || len(harness.store.allocations) != 0 ||
		harness.store.stages[0].AdmittedAt != nil || harness.store.run.State != runstore.RunPending {
		t.Fatalf("raced placement not released = reserves:%d grants:%d cached:%d allocations:%d admitted:%v run:%s",
			harness.allocator.reserveCalls, len(harness.allocator.grants), len(harness.allocator.cached),
			len(harness.store.allocations), harness.store.stages[0].AdmittedAt, harness.store.run.State)
	}

	// A2: on resume the Stage is placed again and runs to success.
	harness.persistence.queuePaused = false
	worked, err = harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked || harness.store.run.State != runstore.RunSucceeded ||
		harness.allocator.reserveCalls != 2 {
		t.Fatalf("resumed RunOnce = (%v, %v), run=%s reserves=%d",
			worked, err, harness.store.run.State, harness.allocator.reserveCalls)
	}
}
