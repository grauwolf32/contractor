package scheduler

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/jackc/pgx/v5"
)

// TestGatewayRecoveryHealthyRouteUpdatesTakeNoLocks holds the Run row and the
// owner's route row locked, as a concurrent commit point or another Run's
// model call would: a model call on the healthy route still acquires and
// reports its outcome, because neither update changes anything and both are
// decided from one read without locks (SCH-R165). Once the route is blocked,
// an acquire takes the locking path again and waits for those locks.
func TestGatewayRecoveryHealthyRouteUpdatesTakeNoLocks(t *testing.T) {
	f := newRecoveryFixture(t)
	participant := f.run(t, "healthy-run", f.route)
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	holdLocks := func() pgx.Tx {
		t.Helper()
		tx, err := f.pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tx.Exec(ctx, `SELECT 1 FROM workflow_runs WHERE run_id = $1 FOR UPDATE`, "healthy-run"); err != nil {
			_ = tx.Rollback(ctx)
			t.Fatal(err)
		}
		if _, err := tx.Exec(ctx, `SELECT 1 FROM gateway_recovery_routes WHERE route_key = $1 FOR UPDATE`, f.route.Key()); err != nil {
			_ = tx.Rollback(ctx)
			t.Fatal(err)
		}
		return tx
	}
	// A locking update would wait for the held rows until this deadline.
	update := func(request gatewayrecovery.Request) (gatewayrecovery.Decision, error) {
		bounded, cancelBounded := context.WithTimeout(ctx, 5*time.Second)
		defer cancelBounded()
		return participant.Update(bounded, request)
	}

	held := holdLocks()
	// Release whichever locks are held when the test ends, also on failure,
	// so the fixture can drop its schema.
	defer func() { _ = held.Rollback(context.Background()) }()
	decision, err := update(gatewayrecovery.Request{RequestID: "healthy-call", Action: "acquire"})
	if err != nil || !decision.Allowed || decision.Code != "" {
		t.Fatalf("acquire on the healthy route = (%+v, %v), want allowed without locks", decision, err)
	}
	for _, action := range []string{"succeeded", "finished", "released"} {
		if _, err := update(gatewayrecovery.Request{RequestID: "healthy-call", Action: action}); err != nil {
			t.Fatalf("%s on the healthy route: %v, want an answer without locks", action, err)
		}
	}
	if err := held.Rollback(ctx); err != nil {
		t.Fatal(err)
	}

	recoveryUpdate(t, participant, "outage", "failed")
	held = holdLocks()
	blocked, cancelBlocked := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancelBlocked()
	decision, err = participant.Update(blocked, gatewayrecovery.Request{RequestID: "blocked-call", Action: "acquire"})
	if err == nil || blocked.Err() == nil {
		t.Fatalf("acquire on the blocked route = (%+v, %v) before its deadline, want it to wait for the Run and route locks",
			decision, err)
	}
}
