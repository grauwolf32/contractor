package runstore

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestPostgresListenerOutlivesQueryBudgetsAndReleasesOnShutdown(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	// The pool and the notifying commits, the test's instruments, keep the
	// default budgets, so a loaded host cannot fail them; only the listener
	// under test runs under the short budgets it must outlive.
	pool, err := persistencepostgres.OpenPool(ctx, url, persistencepostgres.PoolOptions{MaxConnections: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	listener, err := NewPostgresRunEventListener(pool)
	if err != nil {
		t.Fatal(err)
	}
	listenCtx, stop := context.WithCancel(ctx)
	defer stop()
	budgets := persistencepostgres.Budgets{
		AcquireTimeout: 100 * time.Millisecond, QueryTimeout: 300 * time.Millisecond,
		StatementTimeout: 200 * time.Millisecond, LockTimeout: 50 * time.Millisecond,
		IdleTransactionTimeout: 100 * time.Millisecond,
	}
	budgetedListenCtx, err := persistencepostgres.WithOperationBudgets(listenCtx, budgets)
	if err != nil {
		t.Fatal(err)
	}
	const runID = "budget-listener-notification"
	notify := func() error {
		_, err := pool.Exec(ctx, `SELECT pg_notify($1,$2)`, runEventNotificationChannel, runID)
		return err
	}
	received := make(chan string, 1)
	readied := make(chan error, 1)
	done := make(chan error, 1)
	go func() {
		// ready must mean that LISTEN is in effect: a commit made while ready
		// runs is notified, so the Hub catches up only once after ready.
		done <- listener.Listen(budgetedListenCtx, func() { readied <- notify() }, func(id string) {
			if id == runID {
				select {
				case received <- id:
				default:
				}
			}
		})
	}()
	select {
	case err := <-readied:
		if err != nil {
			t.Fatal(err)
		}
	case err := <-done:
		t.Fatalf("listener failed before LISTEN: %v", err)
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	select {
	case <-received:
	case err := <-done:
		t.Fatalf("listener failed: %v", err)
	case <-ctx.Done():
		t.Fatalf("a commit made after ready was not notified: %v", ctx.Err())
	}
	// WaitForNotification must keep the subscription context, not the
	// acquire and query hook contexts. Those end at the latest one query
	// budget after LISTEN started, before ready; twice that is a lower bound
	// a loaded host can only lengthen, so it cannot fail the test.
	timer := time.NewTimer(2 * budgets.QueryTimeout)
	defer timer.Stop()
	select {
	case err := <-done:
		t.Fatalf("listener inherited a short query budget: %v", err)
	case <-timer.C:
	}
	if err := notify(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-received:
	case err := <-done:
		t.Fatalf("listener failed: %v", err)
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	stop()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("shutdown cause: %v", err)
		}
	case <-ctx.Done():
		t.Fatal("listener shutdown did not release its connection before the test deadline")
	}
	if pool.Stat().AcquiredConns() != 0 {
		t.Fatalf("listener leaked acquired connections: %d", pool.Stat().AcquiredConns())
	}
	if _, err := pool.Exec(ctx, `SELECT 1`); err != nil {
		t.Fatalf("pool after listener shutdown: %v", err)
	}
}
