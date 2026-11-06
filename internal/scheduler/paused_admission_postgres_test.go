package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// The owner's queue pauses after placement pinned the Stage's allocation rows
// but before admission. The deferred Stage keeps no rows, no live reservation
// and no recovery probe; after resume it is placed again and succeeds as its
// first attempt, so no retry is consumed (SCH-R152).
func TestPostgresPausedAdmissionPlacesStageAgainAfterResume(t *testing.T) {
	for _, test := range []struct {
		name         string
		blockedRoute bool
	}{
		{name: "healthy model route"},
		// Admission reserves the probe of a blocked, due route for this Run.
		{name: "admission reserved the recovery probe", blockedRoute: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			h := newDeferredPlacementHarness(t, ctx, nil)
			if _, err := h.pool.Exec(ctx, `
UPDATE workflow_runs SET state = 'pending', started_at = NULL, state_reason_code = 'awaiting_admission'
WHERE run_id = 'run-1'`); err != nil {
				t.Fatal(err)
			}
			stage := h.workflow.Stages[h.workflow.EntryStage]
			routes := reservationModelRoutes("user-1", schedulerTestReservations(t, stage)[0])
			if test.blockedRoute {
				for _, route := range routes {
					blockRecoveryRoute(t, h.scheduler.options.GatewayRecovery, route, true)
				}
			}
			setPaused := func(paused bool) {
				t.Helper()
				control, err := h.store.GetOwnerQueueControl(ctx, "user-1")
				if err != nil {
					t.Fatal(err)
				}
				if _, err := h.store.UpdateOwnerQueueControl(ctx, runstore.UpdateOwnerQueueControlParams{
					OwnerID: "user-1", ExpectedRevision: control.Revision, Paused: paused,
				}); err != nil {
					t.Fatal(err)
				}
			}
			allocator := h.workers.allocator
			pin := allocator.pin
			paused := false
			allocator.pin = func(ctx context.Context, reservations []controlplane.Reservation) error {
				if err := pin(ctx, reservations); err != nil {
					return err
				}
				if !paused {
					paused = true

					setPaused(true)
				}
				return nil
			}

			if worked, err := h.scheduler.RunOnce(ctx); !worked || !errors.Is(err, ErrDeferred) {
				t.Fatalf("raced RunOnce = (%v, %v), want deferred", worked, err)
			}
			executions, err := h.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(executions) != 1 || executions[0].State != runstore.StagePreparing ||
				executions[0].AdmittedAt != nil || executions[0].Termination != nil {
				t.Fatalf("deferred StageExecutions = (%v, %v)", stageExecutionStates(executions), err)
			}
			allocations, err := h.store.ListStageAllocations(ctx, executions[0].StageExecutionID)
			if err != nil || len(allocations) != 0 || len(allocator.grants) != 0 || len(allocator.cached) != 0 ||
				allocator.released[executions[0].StageExecutionID] {
				t.Fatalf("deferred placement = rows:%d grants:%d cached:%d tombstone:%v err:%v",
					len(allocations), len(allocator.grants), len(allocator.cached),
					allocator.released[executions[0].StageExecutionID], err)
			}
			// Competing admission can take the relinquished probe immediately.
			if test.blockedRoute {
				service := h.scheduler.options.GatewayRecovery
				if allowed, err := service.Admit(ctx, "competitor", "competitor-stage", routes); err != nil || !allowed {
					t.Fatalf("probe not released: %v %v", allowed, err)
				}
				service.EndStage("competitor", "competitor-stage")
			}

			if run, err := h.store.GetRun(ctx, "run-1"); err != nil || run.State != runstore.RunPending {
				t.Fatalf("deferred Run = (%s, %v), want pending", run.State, err)
			}

			setPaused(false)
			if worked, err := h.scheduler.RunOnce(ctx); err != nil || !worked {
				t.Fatalf("resumed RunOnce = (%v, %v)", worked, err)
			}
			h.requireSingleSucceededStage(t, ctx)
			if executions, err := h.store.ListStageExecutions(ctx, "run-1"); err != nil || executions[0].Attempt != 1 {
				t.Fatalf("resumed Stage attempt = (%v, %v), want the first attempt", stageExecutionStates(executions), err)
			}
			if allocator.reserveCalls != 2 || h.invoker.calls != 1 {
				t.Fatalf("placements=%d Worker calls=%d, want one placement after resume and one call",
					allocator.reserveCalls, h.invoker.calls)
			}
		})
	}
}
