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
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	pool, err := persistencepostgres.OpenPool(ctx, url, persistencepostgres.PoolOptions{
		MaxConnections: 2,
		Budgets:        persistencepostgres.Budgets{AcquireTimeout: 100 * time.Millisecond, QueryTimeout: 300 * time.Millisecond, StatementTimeout: 200 * time.Millisecond, LockTimeout: 50 * time.Millisecond, IdleTransactionTimeout: 100 * time.Millisecond},
	})
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
	const runID = "budget-listener-notification"
	// The notifying commits are the test's instrument, not the listener under
	// test, so a loaded host must not fail them on the short pool budgets.
	notifyCtx, err := persistencepostgres.WithOperationBudgets(ctx, persistencepostgres.Budgets{
		AcquireTimeout: 5 * time.Second, QueryTimeout: 5 * time.Second, StatementTimeout: 4 * time.Second,
		LockTimeout: time.Second, IdleTransactionTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	notify := func() error {
		_, err := pool.Exec(notifyCtx, `SELECT pg_notify($1,$2)`, runEventNotificationChannel, runID)
		return err
	}
	received := make(chan string, 1)
	readied := make(chan error, 1)
	done := make(chan error, 1)
	go func() {
		// ready must mean that LISTEN is in effect: a commit made while ready
		// runs is notified, so the Hub catches up only once after ready.
		done <- listener.Listen(listenCtx, func() { readied <- notify() }, func(id string) {
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
	// LISTEN is outside a transaction, and WaitForNotification must keep the
	// subscription context, not the expired acquire/query hook contexts.
	timer := time.NewTimer(750 * time.Millisecond)
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
	case <-time.After(2 * time.Second):
		t.Fatal("listener shutdown did not release its connection")
	}
	if pool.Stat().AcquiredConns() != 0 {
		t.Fatalf("listener leaked acquired connections: %d", pool.Stat().AcquiredConns())
	}
	if _, err := pool.Exec(ctx, `SELECT 1`); err != nil {
		t.Fatalf("pool after listener shutdown: %v", err)
	}
}
