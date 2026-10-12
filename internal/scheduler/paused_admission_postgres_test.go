package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
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
					if _, err := h.pool.Exec(ctx, `
INSERT INTO gateway_recovery_routes(route_key,owner_id,blocked,failure_code,next_probe_at,automatic_until)
VALUES($1,$2,true,'model_unavailable',clock_timestamp()-interval '1 second',clock_timestamp()+interval '1 hour')`,
						route.Key(), route.OwnerID); err != nil {
						t.Fatal(err)
					}
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
					if test.blockedRoute {
						var reserved int
						if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM gateway_recovery_routes WHERE probe_run_id='run-1'`).Scan(&reserved); err != nil || reserved != len(routes) {
							t.Errorf("admission reserved %d of %d probes: %v", reserved, len(routes), err)
						}
					}
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
			var reserved int
			if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM gateway_recovery_routes WHERE probe_run_id='run-1'`).Scan(&reserved); err != nil || reserved != 0 {
				t.Fatalf("deferred Run kept %d recovery probe reservations: %v", reserved, err)
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

// A second admission of the same Run, as when placement is retried after
// admission reserved the probe of a blocked route, takes that probe again
// instead of being refused by its own reservation. A route the Run no longer
// admits does not keep the reservation.
func TestPostgresGatewayReadmissionReusesOwnProbeReservation(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	service := newProbeTestRecovery(t, pool)
	route := gatewayrecovery.Route{OwnerID: "user-1", GatewayDigest: "gateway", Model: "model"}
	other := route
	other.Model = "other-model"
	if _, err := pool.Exec(ctx, `
INSERT INTO gateway_recovery_routes(route_key,owner_id,blocked,failure_code,next_probe_at,automatic_until)
VALUES($1,$2,true,'model_unavailable',clock_timestamp()-interval '1 second',clock_timestamp()+interval '1 hour')`,
		route.Key(), route.OwnerID); err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	pending := func(runID string) {
		t.Helper()
		if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{
			RunID: runID, OwnerID: "user-1", WorkflowName: "test", WorkflowVersion: "1",
			WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: []byte(`{}`),
			RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
			t.Fatal(err)
		}
	}
	pending("run-readmitted")
	pending("run-competing")
	probeHolder := func() string {
		t.Helper()
		var holder *string
		if err := pool.QueryRow(ctx, `SELECT probe_run_id FROM gateway_recovery_routes WHERE route_key=$1`, route.Key()).Scan(&holder); err != nil {
			t.Fatal(err)
		}
		return stringValue(holder)
	}
	for attempt := 1; attempt <= 2; attempt++ {
		if allowed, err := service.Admit(ctx, "run-readmitted", []gatewayrecovery.Route{route}); err != nil || !allowed {
			t.Fatalf("admission %d of the reserving Run = (%v, %v)", attempt, allowed, err)
		}
		if holder := probeHolder(); holder != "run-readmitted" {
			t.Fatalf("probe after admission %d is reserved by %s", attempt, holder)
		}
	}
	if allowed, err := service.Admit(ctx, "run-competing", []gatewayrecovery.Route{route}); err != nil || allowed {
		t.Fatalf("competing Run admitted while the probe is reserved = (%v, %v)", allowed, err)
	}
	if allowed, err := service.Admit(ctx, "run-readmitted", []gatewayrecovery.Route{other}); err != nil || !allowed {
		t.Fatalf("admission on another route = (%v, %v)", allowed, err)
	}
	if holder := probeHolder(); holder != "<none>" {
		t.Fatalf("dropped route kept the probe reservation of %s", holder)
	}
	if allowed, err := service.Admit(ctx, "run-competing", []gatewayrecovery.Route{route}); err != nil || !allowed {
		t.Fatalf("competing Run after the reservation was dropped = (%v, %v)", allowed, err)
	}
}
