package evalservice

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// waitUntilBlockedBy waits until another backend waits for a lock that the
// holder backend keeps. The batch must not finish before it is blocked.
func waitUntilBlockedBy(t *testing.T, pool *pgxpool.Pool, holder int, done <-chan error) {
	t.Helper()
	for {
		var blocked bool
		if err := pool.QueryRow(t.Context(), `
SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE $1 = ANY(pg_blocking_pids(pid)))`, holder).Scan(&blocked); err != nil {
			t.Fatal(err)
		}
		if blocked {
			return
		}
		select {
		case err := <-done:
			t.Fatalf("batch finished before the stop committed: %v", err)
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// A stop commits after a native batch chose its members from the running
// experiment but before the batch locks the experiment to admit them.
func TestPostgresStopDuringBatchAdmitsNothingAfterTransition(t *testing.T) {
	for _, stop := range []struct {
		kind evaldomain.CommandKind
		want evaldomain.State
	}{
		{kind: evaldomain.CommandPause, want: evaldomain.StatePaused},
		{kind: evaldomain.CommandCancel, want: evaldomain.StateCancelled},
	} {
		t.Run(string(stop.kind), func(t *testing.T) {
			h := newHarness(t)
			e := h.preparedWithCapacity(t, "workflow", 3)
			h.command(t, e, "start")
			store := evalstore.NewPostgresStore(h.pool)
			claims, err := store.Claim(t.Context(), "batch-controller", time.Minute, 1)
			if err != nil || len(claims) != 1 {
				t.Fatalf("claim: %v %v", claims, err)
			}
			// Acknowledge Start first, so admission takes the tick's first experiment lock.
			if err = h.service.recoverCommands(t.Context(), h.get(t, e.ID), claims[0]); err != nil {
				t.Fatal(err)
			}
			plan, err := store.FrozenPlan(t.Context(), e.OwnerID, e.ID)
			if err != nil {
				t.Fatal(err)
			}
			e = h.get(t, e.ID)
			command := evaldomain.Command{Kind: stop.kind, PlanSHA256: plan.SHA256}
			doc := frozen(t, "Command", command)
			tx, err := h.pool.Begin(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = tx.Rollback(context.Background()) }()
			if _, err = evalstore.NewTxStore(tx).Command(t.Context(), evalstore.CommandParams{
				Scope: h.scope, ExperimentID: e.ID, CommandID: "stop-during-batch",
				Command: command, Mutation: identity(t, "stop-during-batch", e.Revision, doc),
			}); err != nil {
				t.Fatal(err)
			}
			var stopper int
			if err = tx.QueryRow(t.Context(), `SELECT pg_backend_pid()`).Scan(&stopper); err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() {
				_, err := h.service.Tick(t.Context(), claims[0])
				done <- err
			}()
			waitUntilBlockedBy(t, h.pool, stopper, done)
			if err = tx.Commit(t.Context()); err != nil {
				t.Fatal(err)
			}
			var domain *evaldomain.Error
			if err = <-done; err != nil && (!errors.As(err, &domain) || domain.Code != "eval_not_ready") {
				t.Fatalf("batch after %s: %v", stop.kind, err)
			}
			if n := count(t, h.pool, "eval_submissions"); n != 0 {
				t.Fatalf("batch admitted %d members after %s committed", n, stop.kind)
			}
			if err = store.ReleaseClaim(t.Context(), claims[0]); err != nil {
				t.Fatal(err)
			}
			c := h.coordinator(t, "controller")
			for n := 0; n < 3; n++ {
				tick(t, c)
			}
			e = h.get(t, e.ID)
			if e.State != stop.want || e.Outstanding != 0 || count(t, h.pool, "eval_submissions") != 0 {
				t.Fatalf("%s drained to %s with outstanding=%d submissions=%d", stop.kind, e.State, e.Outstanding, count(t, h.pool, "eval_submissions"))
			}
		})
	}
}
