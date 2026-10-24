package app

import (
	"context"
	"io"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// The lease session is a copy of the working pool's connection settings, so
// on a real PostgreSQL server every startup lease step runs under the
// configured database budgets, or under an earlier caller deadline, and the
// session carries the configured server-side timeouts. After startup the
// tolerance window bounds a standby acquisition probe when it ends first,
// and the liveness probe runs outside the client query budget (PER-R27,
// PER-R28).
func TestPostgresControlPlaneLeaseSessionInheritsStartupBudgets(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	// Distinct values, none of them a default, so each one is traceable.
	settings := DatabaseSettings{
		ConnectTimeout: 2 * time.Minute, AcquireTimeout: 3 * time.Second, QueryTimeout: 3 * time.Minute,
		StatementTimeout: 150 * time.Second, LockTimeout: 7 * time.Second, IdleTransactionTimeout: 4 * time.Minute,
	}
	config, err := persistencepostgres.PoolConfig(databaseURL, settings.poolOptions(nil))
	if err != nil {
		t.Fatal(err)
	}
	recorder := recordStartupBudgets(t, config)
	pool, err := pgxpool.NewWithConfig(t.Context(), config)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	// The process context has no deadline: the configured budgets own the
	// connection, the keepalive configuration, the first acquisition, the
	// identification of the session that took the lease and the holder
	// lookup.
	ctx := t.Context()
	if _, ok := ctx.Deadline(); ok {
		t.Fatal("the startup context must have no deadline of its own")
	}
	before := time.Now()
	active, err := openControlPlaneLease(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	defer active.Close()
	if _, err := active.tryAcquire(ctx); err != nil {
		t.Fatal(err)
	}
	active.holderPID(ctx)
	after := time.Now()
	assertBudgetDeadlines(t, "lease session connect", recorder.dialsSince(0), settings.ConnectTimeout, before, after)
	startup := recorder.queriesSince(0)
	if len(startup) != 5 {
		t.Fatalf("recorded %d startup statements, want keepalives, acquisition, the two of the "+
			"holder's identification and holder lookup", len(startup))
	}
	assertBudgetDeadlines(t, "startup lease statement", startup, settings.QueryTimeout, before, after)

	// The handover runs on the working pool: database.queryTimeout bounds
	// each of its statements, and the session's lock_timeout, checked on
	// the lease session below from the same startup parameters, its wait
	// (PER-R34).
	traced := len(recorder.queriesSince(0))
	before = time.Now()
	if err := active.awaitHandover(ctx); err != nil {
		t.Fatal(err)
	}
	after = time.Now()
	if handover := recorder.queriesSince(traced); len(handover) == 0 {
		t.Fatal("the lease handover ran no statement")
	} else {
		assertBudgetDeadlines(t, "lease handover statement", handover, settings.QueryTimeout, before, after)
	}

	session := active.conn.Config()
	if session.ConnectTimeout != settings.ConnectTimeout || session.Tracer != pool.Config().ConnConfig.Tracer {
		t.Fatal("the lease session does not carry the working pool's connect timeout and budget tracer")
	}
	rows, err := active.conn.Query(ctx, `SELECT name, setting FROM pg_settings
WHERE name IN ('statement_timeout', 'lock_timeout', 'idle_in_transaction_session_timeout')`)
	if err != nil {
		t.Fatal(err)
	}
	serverSide := map[string]string{}
	for rows.Next() {
		var name, setting string
		if err := rows.Scan(&name, &setting); err != nil {
			t.Fatal(err)
		}
		serverSide[name] = setting
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{
		"statement_timeout": "150000", "lock_timeout": "7000", "idle_in_transaction_session_timeout": "240000",
	} {
		if serverSide[name] != want {
			t.Fatalf("lease session %s = %q ms, want the configured %s ms", name, serverSide[name], want)
		}
	}

	// The active liveness probe is a protocol-level ping: the client query
	// budget never shortens it, only the tolerance window does.
	traced = len(recorder.queriesSince(0))
	if err := active.conn.Ping(ctx); err != nil {
		t.Fatal(err)
	}
	if len(recorder.queriesSince(traced)) != 0 {
		t.Fatal("the liveness probe ran under the client query budget")
	}

	// An earlier caller deadline is kept exactly by every startup step of a
	// second Server, which finds the lease held and looks up its holder.
	callerCtx, cancel := context.WithDeadline(ctx, time.Now().Add(time.Minute))
	defer cancel()
	dials, traced := len(recorder.dialsSince(0)), len(recorder.queriesSince(0))
	standby, err := openControlPlaneLease(callerCtx, pool)
	if err != nil {
		t.Fatal(err)
	}
	defer standby.Close()
	if _, err := standby.tryAcquire(callerCtx); err != nil {
		t.Fatal(err)
	}
	standby.holderPID(callerCtx)
	assertCallerDeadlines(t, "lease session connect", recorder.dialsSince(dials), callerCtx)
	if steps := recorder.queriesSince(traced); len(steps) != 3 {
		t.Fatalf("recorded %d startup statements under the caller deadline, want 3", len(steps))
	} else {
		assertCallerDeadlines(t, "startup lease statement", steps, callerCtx)
	}

	// After startup the tolerance window bounds each standby acquisition
	// probe and each reconnect of its lock-free session when it ends before
	// the inherited budgets.
	standby.tolerance = 90 * time.Second
	traced = len(recorder.queriesSince(0))
	before = time.Now()
	if _, err := standby.acquireOrReconnect(ctx, slog.New(slog.NewTextHandler(io.Discard, nil))); err != nil {
		t.Fatal(err)
	}
	after = time.Now()
	assertBudgetDeadlines(t, "standby acquisition probe", recorder.queriesSince(traced), standby.tolerance, before, after)
	dials, traced = len(recorder.dialsSince(0)), len(recorder.queriesSince(0))
	before = time.Now()
	if err := standby.reconnect(ctx); err != nil {
		t.Fatal(err)
	}
	after = time.Now()
	assertBudgetDeadlines(t, "standby reconnect", recorder.dialsSince(dials), standby.tolerance, before, after)
	assertBudgetDeadlines(t, "standby reconnect keepalives", recorder.queriesSince(traced), standby.tolerance, before, after)
}
