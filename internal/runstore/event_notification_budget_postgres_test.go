package runstore

import (
	"context"
	"crypto/rand"
	"errors"
	"os"
	"strconv"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresListenerOutlivesQueryBudgetsAndReleasesOnShutdown(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	// Client-side budgets allow a loaded host to connect and execute LISTEN;
	// the dedicated listener session really carries short server-side budgets.
	budgets := persistencepostgres.Budgets{
		AcquireTimeout: time.Minute, QueryTimeout: time.Minute,
		StatementTimeout: 200 * time.Millisecond, LockTimeout: 50 * time.Millisecond,
		IdleTransactionTimeout: 100 * time.Millisecond,
	}
	config, err := persistencepostgres.PoolConfig(url, persistencepostgres.PoolOptions{MaxConnections: 2, Budgets: budgets})
	if err != nil {
		t.Fatal(err)
	}
	// Check the actual LISTEN backend, after LISTEN has completed, rather
	// than assuming another connection's settings represent the subscription.
	trace := &listenerBudgetTrace{query: config.ConnConfig.Tracer, t: t, budgets: budgets, checked: make(chan struct{}, 1)}
	config.ConnConfig.Tracer = trace
	pool, err := pgxpool.NewWithConfig(ctx, config)
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
	notifier, err := persistencepostgres.OpenPool(ctx, url, persistencepostgres.PoolOptions{MaxConnections: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer notifier.Close()
	runID := "budget-listener-" + rand.Text()
	notify := func() error {
		_, err := notifier.Exec(ctx, `SELECT pg_notify($1,$2)`, runEventNotificationChannel, runID)
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
	case <-trace.checked:
	default:
		t.Fatal("LISTEN readiness did not execute the session-budget probe")
	}
	select {
	case <-received:
	case err := <-done:
		t.Fatalf("listener failed: %v", err)
	case <-ctx.Done():
		t.Fatalf("a commit made after ready was not notified: %v", ctx.Err())
	}
	// WaitForNotification must keep the subscription context, not the
	// acquire and query hook contexts. The pool's server-side budgets are
	// shorter than this wait; elapsed time is only a lower bound here.
	timer := time.NewTimer(2 * budgets.StatementTimeout)
	defer timer.Stop()
	select {
	case err := <-done:
		t.Fatalf("listener stopped between notifications under short server budgets: %v", err)
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

// listenerBudgetTrace checks the same connection that executed LISTEN. Use
// the wire connection directly so inspecting settings cannot recurse into
// pgx's query tracer. The ordinary budget tracer still wraps each operation.
type listenerBudgetTrace struct {
	query   pgx.QueryTracer
	t       *testing.T
	budgets persistencepostgres.Budgets
	checked chan struct{}
}

func (h *listenerBudgetTrace) TraceQueryStart(ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	return h.query.TraceQueryStart(ctx, conn, data)
}
func (h *listenerBudgetTrace) TraceQueryEnd(ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryEndData) {
	h.query.TraceQueryEnd(ctx, conn, data)
	if data.Err != nil || data.CommandTag.String() != "LISTEN" {
		return
	}
	defer func() { h.checked <- struct{}{} }()
	if ctx.Err() == nil {
		h.t.Error("the LISTEN query hook did not end its operation context")
	}
	results, err := conn.PgConn().Exec(h.t.Context(), `SELECT
(extract(epoch FROM current_setting('statement_timeout')::interval)*1000)::bigint,
(extract(epoch FROM current_setting('lock_timeout')::interval)*1000)::bigint,
(extract(epoch FROM current_setting('idle_in_transaction_session_timeout')::interval)*1000)::bigint`).ReadAll()
	if err != nil {
		h.t.Error(err)
		return
	}
	if len(results) != 1 || len(results[0].Rows) != 1 || len(results[0].Rows[0]) != 3 {
		h.t.Error("cannot inspect LISTEN session budgets")
		return
	}
	for index, want := range []time.Duration{h.budgets.StatementTimeout, h.budgets.LockTimeout, h.budgets.IdleTransactionTimeout} {
		if string(results[0].Rows[0][index]) != strconv.FormatInt(want.Milliseconds(), 10) {
			h.t.Error("LISTEN connection does not carry the short server budgets")
		}
	}
}

func (h *listenerBudgetTrace) TraceAcquireStart(ctx context.Context, pool *pgxpool.Pool, data pgxpool.TraceAcquireStartData) context.Context {
	return h.query.(pgxpool.AcquireTracer).TraceAcquireStart(ctx, pool, data)
}
func (h *listenerBudgetTrace) TraceAcquireEnd(ctx context.Context, pool *pgxpool.Pool, data pgxpool.TraceAcquireEndData) {
	h.query.(pgxpool.AcquireTracer).TraceAcquireEnd(ctx, pool, data)
}
