package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type discardResponseLoss struct{ AtomicPersistence }

func (p discardResponseLoss) DiscardUnadmittedPlacement(ctx context.Context, runID, stageID string, discard func() error) error {
	if err := p.AtomicPersistence.DiscardUnadmittedPlacement(ctx, runID, stageID, discard); err != nil {
		return err
	}
	return errors.New("injected lost discard acknowledgement")
}

// A definite COMMIT failure after the live discard and a lost success response
// both reconcile through the same unadmitted Stage; neither spends a retry.
func TestPostgresPauseDiscardRecoversCommitFailureWithoutAnotherAttempt(t *testing.T) {
	for _, failure := range []string{"rollback", "response lost"} {
		t.Run(failure, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			h := newDeferredPlacementHarness(t, ctx, nil)
			if _, err := h.pool.Exec(ctx, `UPDATE workflow_runs SET state='pending',started_at=NULL,state_reason_code='awaiting_admission' WHERE run_id='run-1'`); err != nil {
				t.Fatal(err)
			}
			if failure == "rollback" {
				if _, err := h.pool.Exec(ctx, `
CREATE FUNCTION test_fail_discard_commit() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'injected commit failure' USING ERRCODE='40001'; END;
$$;
CREATE CONSTRAINT TRIGGER test_fail_discard_commit AFTER DELETE ON stage_allocations
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION test_fail_discard_commit()`); err != nil {
					t.Fatal(err)
				}
			} else {
				h.scheduler.persistence = discardResponseLoss{h.scheduler.persistence}
			}
			allocator := h.workers.allocator
			pin, paused := allocator.pin, false
			allocator.pin = func(c context.Context, reservations []controlplane.Reservation) error {
				if err := pin(c, reservations); err != nil {
					return err
				}
				if !paused {
					paused = true
					_, err := h.store.UpdateOwnerQueueControl(c, runstore.UpdateOwnerQueueControlParams{OwnerID: "user-1", ExpectedRevision: 0, Paused: true})
					return err
				}
				return nil
			}
			if worked, err := h.scheduler.RunOnce(ctx); !worked || !errors.Is(err, ErrDeferred) {
				t.Fatalf("pause = %v %v", worked, err)
			}
			stage, err := h.store.GetActiveStageExecution(ctx, "run-1")
			if err != nil {
				t.Fatal(err)
			}
			allocations, err := h.store.ListStageAllocations(ctx, stage.StageExecutionID)
			if err != nil {
				t.Fatal(err)
			}
			want := 0
			if failure == "rollback" {
				want = 1
			}
			if len(allocations) != want || len(allocator.grants) != 0 || h.invoker.calls != 0 {
				t.Fatalf("fault setup rows=%d grants=%d calls=%d", len(allocations), len(allocator.grants), h.invoker.calls)
			}
			if failure == "rollback" {
				if _, err := h.pool.Exec(ctx, `DROP TRIGGER test_fail_discard_commit ON stage_allocations`); err != nil {
					t.Fatal(err)
				}
			}
			// Reconcile while still paused, without reserving any new slot or work.
			if _, err := h.scheduler.RunOnce(ctx); err != nil && !errors.Is(err, ErrDeferred) {
				t.Fatal(err)
			}
			allocations, err = h.store.ListStageAllocations(ctx, stage.StageExecutionID)
			if err != nil || len(allocations) != 0 || len(allocator.grants) != 0 || h.invoker.calls != 0 {
				t.Fatalf("paused reconciliation rows=%d grants=%d calls=%d err=%v", len(allocations), len(allocator.grants), h.invoker.calls, err)
			}
			control, err := h.store.GetOwnerQueueControl(ctx, "user-1")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := h.store.UpdateOwnerQueueControl(ctx, runstore.UpdateOwnerQueueControlParams{OwnerID: "user-1", ExpectedRevision: control.Revision, Paused: false}); err != nil {
				t.Fatal(err)
			}
			if worked, err := h.scheduler.RunOnce(ctx); err != nil || !worked {
				t.Fatalf("resume = %v %v", worked, err)
			}
			h.requireSingleSucceededStage(t, ctx)
			current, err := h.store.GetStageExecution(ctx, stage.StageExecutionID)
			if err != nil || current.Attempt != 1 || current.Termination != nil || h.invoker.calls != 1 {
				t.Fatalf("resumed Stage attempt=%d termination=%+v calls=%d err=%v", current.Attempt, current.Termination, h.invoker.calls, err)
			}
			var probes int
			if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM gateway_recovery_routes WHERE probe_run_id='run-1'`).Scan(&probes); err != nil || probes != 0 {
				t.Fatalf("recovery probe reservations=%d err=%v", probes, err)
			}
		})
	}
}
