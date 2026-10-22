package scheduler

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// A suspended host can advance PostgreSQL's wall clock past a Run lease while
// the original Planner and its renewal timer remain alive. Force that ordering
// without sleeping: park the Worker and renewal, expire the SQL lease, and let
// another lane try to claim it before the original lane renews.
func TestPostgresLivePlannerSurvivesExpiredClaimBeforeRenewal(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	h := newDeferredPlacementHarness(t, ctx, nil)
	clock := &manualRenewalClock{now: time.Now().UTC(), waits: make(chan chan time.Time, 4)}
	h.scheduler.options.Clock = clock
	started, finish := make(chan struct{}), make(chan struct{})
	h.invoker.before = func(context.Context, contracts.WorkerHandle) error {
		close(started)
		<-finish
		return nil
	}
	done := make(chan error, 1)
	joined := make(chan struct{})
	go func() {
		defer close(joined)
		_, err := h.scheduler.progressOneClaim(ctx)
		done <- err
	}()
	finished := false
	defer func() {
		cancel()
		if !finished {
			close(finish)
		}
		select {
		case <-joined:
		case <-time.After(5 * time.Second):
			t.Error("original lane did not join")
		}
	}()
	select {
	case <-started:
	case <-ctx.Done():
		t.Fatal("Planner did not dispatch its Worker", ctx.Err())
	}
	original, err := h.store.GetRun(ctx, "run-1")
	if err != nil || original.SchedulerClaim == nil {
		t.Fatalf("original Run claim = %+v, %v", original.SchedulerClaim, err)
	}
	if _, err := h.pool.Exec(ctx, `UPDATE workflow_runs
SET scheduler_claimed_at = clock_timestamp() - interval '3 minutes',
    scheduler_claim_expires_at = clock_timestamp() - interval '1 minute'
WHERE run_id = 'run-1'`); err != nil {
		t.Fatal(err)
	}
	if worked, err := h.scheduler.progressOneClaim(ctx); worked || err != nil {
		t.Fatalf("competing lane re-entered the live Planner: worked=%v err=%v", worked, err)
	}
	current, err := h.store.GetRun(ctx, "run-1")
	if err != nil || current.State != runstore.RunRunning || current.SchedulerClaim == nil ||
		current.SchedulerClaim.ClaimID != original.SchedulerClaim.ClaimID {
		t.Fatalf("competing lane replaced the live claim: state=%s claim=%+v err=%v", current.State, current.SchedulerClaim, err)
	}
	select {
	case tick := <-clock.waits:
		tick <- clock.now
	case <-ctx.Done():
		t.Fatal("claim renewal did not start", ctx.Err())
	}
	select {
	case <-clock.waits: // The original claim has renewed before the next wait.
	case <-ctx.Done():
		t.Fatal("original claim did not renew", ctx.Err())
	}
	close(finish)
	finished = true
	if err := receiveError(t, done, 5*time.Second, "original Planner completion"); err != nil {
		t.Fatal(err)
	}
	current, err = h.store.GetRun(ctx, "run-1")
	stages, stagesErr := h.store.ListStageExecutions(ctx, "run-1")
	if err != nil || stagesErr != nil || current.State != runstore.RunSucceeded ||
		len(stages) != 1 || stages[0].Termination != nil || h.invoker.calls != 1 {
		t.Fatalf("original completion = state=%s stages=%+v calls=%d errors=%v/%v", current.State, stages, h.invoker.calls, err, stagesErr)
	}
}
