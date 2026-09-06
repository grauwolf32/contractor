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
	received := make(chan string, 1)
	done := make(chan error, 1)
	go func() {
		done <- listener.Listen(listenCtx, func(id string) {
			if id == runID {
				select {
				case received <- id:
				default:
				}
			}
		})
	}()
	// LISTEN is outside a transaction, and WaitForNotification must keep the
	// subscription context, not the expired acquire/query hook contexts.
	timer := time.NewTimer(750 * time.Millisecond)
	defer timer.Stop()
	select {
	case err := <-done:
		t.Fatalf("listener inherited a short query budget: %v", err)
	case <-timer.C:
	}
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
notifications:
	for {
		if _, err := pool.Exec(ctx, `SELECT pg_notify($1,$2)`, runEventNotificationChannel, runID); err != nil {
			t.Fatal(err)
		}
		select {
		case <-received:
			break notifications
		case err := <-done:
			t.Fatalf("listener failed: %v", err)
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-tick.C:
		}
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
