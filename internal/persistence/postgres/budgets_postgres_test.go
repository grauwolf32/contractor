package postgres

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func smallDatabaseBudgets() Budgets {
	return Budgets{100 * time.Millisecond, 500 * time.Millisecond, 300 * time.Millisecond, 75 * time.Millisecond, 200 * time.Millisecond}
}

// ordinaryWaitBudgets is the pool policy whose server-side budgets
// TestPostgresOrdinaryWaitBudgetsAndRecovery proves by the SQLSTATE each one
// raises. They are far apart, so machine load cannot change which budget ends
// an operation: the lock timeout fires long before the statement timeout,
// whose error reaches the client long before the client-side query timeout,
// and a transaction is idle between its statements far less than the idle
// timeout.
func ordinaryWaitBudgets() Budgets {
	return Budgets{
		AcquireTimeout: 100 * time.Millisecond, QueryTimeout: time.Minute, StatementTimeout: time.Second,
		LockTimeout: 75 * time.Millisecond, IdleTransactionTimeout: 2 * time.Second,
	}
}

// clientBudgetOnly is a pool policy whose client-side query timeout is the
// only budget a statement can reach once its transaction disables the
// server's statement timeout.
func clientBudgetOnly() Budgets {
	return Budgets{
		AcquireTimeout: 100 * time.Millisecond, QueryTimeout: 500 * time.Millisecond, StatementTimeout: 300 * time.Millisecond,
		LockTimeout: 75 * time.Millisecond, IdleTransactionTimeout: 2 * time.Second,
	}
}

// requireSessionBudgets checks the server-side budgets that db's session
// applies.
func requireSessionBudgets(t *testing.T, ctx context.Context, db DBTX, want Budgets) {
	t.Helper()
	var statement, lock, idle int64
	if err := db.QueryRow(ctx, `SELECT (extract(epoch FROM current_setting('statement_timeout')::interval)*1000)::bigint,
(extract(epoch FROM current_setting('lock_timeout')::interval)*1000)::bigint,
(extract(epoch FROM current_setting('idle_in_transaction_session_timeout')::interval)*1000)::bigint`).Scan(&statement, &lock, &idle); err != nil {
		t.Fatal(err)
	}
	if statement != want.StatementTimeout.Milliseconds() || lock != want.LockTimeout.Milliseconds() || idle != want.IdleTransactionTimeout.Milliseconds() {
		t.Fatalf("settings=%d/%d/%d want %+v", statement, lock, idle, want)
	}
}

func budgetTestDatabaseURL(t *testing.T) string {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	return url
}

func budgetTestPool(t *testing.T, ctx context.Context, base *pgxpool.Pool, options PoolOptions) *pgxpool.Pool {
	t.Helper()
	// The helper-generated schema identifier is safe in keyword syntax. The
	// original URL remains opaque and never appears in diagnostics.
	connectionString := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	config, err := pgxpool.ParseConfig(connectionString)
	if err != nil {
		t.Fatal("invalid test connection configuration")
	}
	config.ConnConfig.RuntimeParams["search_path"] = base.Config().ConnConfig.RuntimeParams["search_path"]
	// ConnString preserves the original string, not subsequent config edits.
	// pgx accepts URI query parameters for our test URL.
	separator := "?"
	if strings.Contains(connectionString, "?") {
		separator = "&"
	}
	if !strings.HasPrefix(connectionString, "postgres://") && !strings.HasPrefix(connectionString, "postgresql://") {
		connectionString += " search_path=" + config.ConnConfig.RuntimeParams["search_path"]
	} else {
		connectionString += separator + "search_path=" + config.ConnConfig.RuntimeParams["search_path"]
	}
	pool, err := OpenPool(ctx, connectionString, options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return pool
}

// TestPostgresOrdinaryWaitBudgetsAndRecovery proves each ordinary budget by
// the outcome only that budget produces, never by elapsed time: the pool's
// session settings, the SQLSTATE of a server-side timeout, a client deadline
// where the server's timeouts cannot fire, and a reaped idle transaction. Each
// wait outlasts its budget, so a budget that does not apply fails the test;
// go test -timeout bounds a wait that nothing ends.
func TestPostgresOrdinaryWaitBudgetsAndRecovery(t *testing.T) {
	ctx := t.Context()
	base, _ := isolatedPools(t, ctx, budgetTestDatabaseURL(t))
	var logs bytes.Buffer
	pool := budgetTestPool(t, ctx, base, PoolOptions{MaxConnections: 1, Budgets: ordinaryWaitBudgets(), Logger: slog.New(slog.NewJSONHandler(&logs, nil))})
	requireSessionBudgets(t, ctx, pool, ordinaryWaitBudgets())
	if _, err := pool.Exec(ctx, `CREATE TABLE budget_rows (id integer PRIMARY KEY,value integer NOT NULL); INSERT INTO budget_rows VALUES(1,0)`); err != nil {
		t.Fatal(err)
	}
	assertReusable := func(pool *pgxpool.Pool) {
		t.Helper()
		var value int
		if err := pool.QueryRow(ctx, `SELECT value FROM budget_rows WHERE id=1`).Scan(&value); err != nil || value != 0 {
			t.Fatalf("pool/rollback not reusable: value=%d error=%v", value, err)
		}
	}
	t.Run("exhausted-pool", func(t *testing.T) {
		held, err := pool.Acquire(ctx)
		if err != nil {
			t.Fatal(err)
		}
		_, err = pool.Exec(context.Background(), `SELECT 'secret-sql-argument'`)
		held.Release()
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("unbounded acquire cause: %v", err)
		}
		if pool.Stat().CanceledAcquireCount() != 1 {
			t.Fatalf("cancelled acquires=%d", pool.Stat().CanceledAcquireCount())
		}
		if !strings.Contains(logs.String(), "PostgreSQL pool acquire pressure") || !strings.Contains(logs.String(), "acquired_connections") || strings.Contains(logs.String(), "secret") || strings.Contains(logs.String(), "postgres://") {
			t.Fatalf("unsafe/missing pressure diagnostics: %s", logs.String())
		}
		assertReusable(pool)
	})
	t.Run("statement-abort-no-retry", func(t *testing.T) {
		calls := 0
		err := InTxWithRetry(context.Background(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			calls++
			if _, err := tx.Exec(context.Background(), `UPDATE budget_rows SET value=1`); err != nil {
				return err
			}
			_, err := tx.Exec(context.Background(), `SELECT pg_sleep(30)`)
			return err
		})
		if SQLState(err) != SQLStateQueryCanceled || calls != 1 || IsTransactionConflict(err) {
			t.Fatalf("statement timeout: state=%s attempts=%d error=%v", SQLState(err), calls, err)
		}
		assertReusable(pool)
	})
	t.Run("row-lock-abort-no-retry", func(t *testing.T) {
		locker, err := base.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer locker.Rollback(ctx)
		if _, err := locker.Exec(ctx, `SELECT 1 FROM budget_rows WHERE id=1 FOR UPDATE`); err != nil {
			t.Fatal(err)
		}
		calls := 0
		err = InTxWithRetry(context.Background(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			calls++
			_, err := tx.Exec(context.Background(), `UPDATE budget_rows SET value=2 WHERE id=1`)
			return err
		})
		if SQLState(err) != SQLStateLockNotAvailable || calls != 1 || IsTransactionConflict(err) {
			t.Fatalf("lock timeout: state=%s attempts=%d error=%v", SQLState(err), calls, err)
		}
		if err := locker.Rollback(ctx); err != nil {
			t.Fatal(err)
		}
		assertReusable(pool)
	})
	t.Run("client-budget-even-with-server-timeout-disabled", func(t *testing.T) {
		client := budgetTestPool(t, ctx, base, PoolOptions{MaxConnections: 1, Budgets: clientBudgetOnly()})
		err := InTx(context.Background(), client, pgx.TxOptions{}, func(tx pgx.Tx) error {
			if _, err := tx.Exec(context.Background(), `SET LOCAL statement_timeout=0`); err != nil {
				return err
			}
			if _, err := tx.Exec(context.Background(), `UPDATE budget_rows SET value=3`); err != nil {
				return err
			}
			_, err := tx.Exec(context.Background(), `SELECT pg_sleep(30)`)
			return err
		})
		if !errors.Is(err, context.DeadlineExceeded) || IsTransactionConflict(err) {
			t.Fatalf("client timeout identity: %v", err)
		}
		assertReusable(client)
	})
	t.Run("caller-cancellation-and-rollback", func(t *testing.T) {
		short, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
		defer cancel()
		err := InTx(short, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			if _, err := tx.Exec(short, `UPDATE budget_rows SET value=4`); err != nil {
				return err
			}
			_, err := tx.Exec(short, `SELECT pg_sleep(3)`)
			return err
		})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("caller cancellation lost: %v", err)
		}
		assertReusable(pool)
	})
	t.Run("idle-transaction-is-reaped", func(t *testing.T) {
		tx, err := pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer tx.Rollback(ctx)
		pid := tx.Conn().PgConn().PID()
		if _, err := tx.Exec(ctx, `UPDATE budget_rows SET value=5`); err != nil {
			t.Fatal(err)
		}
		// Only the idle timeout ends this session; without it the loop
		// waits for go test -timeout.
		tick := time.NewTicker(10 * time.Millisecond)
		defer tick.Stop()
		for {
			var exists bool
			if err := base.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=$1)`, pid).Scan(&exists); err != nil {
				t.Fatal(err)
			}
			if !exists {
				break
			}
			<-tick.C
		}
		_ = tx.Rollback(ctx)
		assertReusable(pool)
	})
}

// TestPostgresMaintenanceBudgetsRestorePoolSettings proves the budgets
// through session settings; go test -timeout bounds it, so applying every
// migration takes as long as the machine needs.
func TestPostgresMaintenanceBudgetsRestorePoolSettings(t *testing.T) {
	ctx := t.Context()
	base, _ := isolatedPools(t, ctx, budgetTestDatabaseURL(t))
	pool := budgetTestPool(t, ctx, base, PoolOptions{MaxConnections: 1, Budgets: smallDatabaseBudgets()})
	requireSessionBudgets(t, ctx, pool, smallDatabaseBudgets())
	migration := func(budgets MigrationBudgets) func(context.Context) (context.Context, context.CancelFunc) {
		return func(ctx context.Context) (context.Context, context.CancelFunc) {
			ctx, cancel, err := WithMigrationBudgets(ctx, budgets)
			if err != nil {
				t.Fatal(err)
			}
			return ctx, cancel
		}
	}
	defaultMigration := migration(MigrationBudgets{})
	configuredMigration := migration(MigrationBudgets{StatementTimeout: 3 * time.Minute, LockTimeout: 20 * time.Second})
	for _, budget := range []func(context.Context) (context.Context, context.CancelFunc){defaultMigration, configuredMigration, WithCleanupBudget} {
		for _, rollback := range []bool{false, true} {
			maintenance, done := budget(ctx)
			err := InTx(maintenance, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				requireSessionBudgets(t, maintenance, tx, maintenance.Value(maintenanceBudgetKey{}).(Budgets))
				if _, err := tx.Exec(maintenance, `SELECT pg_sleep(0.65)`); err != nil {
					return err
				}
				if rollback {
					return errors.New("injected rollback")
				}
				return nil
			})
			done()
			if (!rollback && err != nil) || (rollback && (err == nil || err.Error() != "injected rollback")) {
				t.Fatalf("maintenance outcome: %v", err)
			}
			requireSessionBudgets(t, ctx, pool, smallDatabaseBudgets())
		}
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{StatementTimeout: time.Second, LockTimeout: time.Second}); err == nil {
		t.Fatal("invalid migration budgets were accepted")
	}
	if result, err := ApplyMigrations(ctx, pool, MigrationBudgets{StatementTimeout: 3 * time.Minute, LockTimeout: 20 * time.Second}); err != nil || len(result.AppliedVersions) != 0 {
		t.Fatalf("migration replay: %+v %v", result, err)
	}
	requireSessionBudgets(t, ctx, pool, smallDatabaseBudgets())
}

// TestPostgresMigrationLockCancellationLeavesCapacityReusable cancels a
// migrator once it is observed waiting for its leader, not after a timer, and
// then applies every migration through the same one-connection pool; go test
// -timeout bounds the test.
func TestPostgresMigrationLockCancellationLeavesCapacityReusable(t *testing.T) {
	ctx := t.Context()
	base, _ := isolatedPools(t, ctx, budgetTestDatabaseURL(t))
	pool := budgetTestPool(t, ctx, base, PoolOptions{MaxConnections: 1})
	connection, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	migrator := int32(connection.Conn().PgConn().PID())
	connection.Release()
	locker, err := base.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer locker.Rollback(ctx)
	if _, err := locker.Exec(ctx, `SELECT pg_advisory_xact_lock(`+migrationLockKeySQL+`)`, migrationLockKey); err != nil {
		t.Fatal(err)
	}
	waiting, stop := context.WithCancel(ctx)
	defer stop()
	done := make(chan error, 1)
	go func() {
		_, err := ApplyMigrations(waiting, pool, MigrationBudgets{})
		done <- err
	}()
	awaitMigrationLeaderPoll(t, ctx, base, migrator, done)
	stop()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("migration lock cancellation: %v", err)
	}
	if err := locker.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatalf("migration after cancellation: %v", err)
	}
}

// awaitMigrationLeaderPoll waits until the backend pid has polled for the
// migration lock, which the test holds, so its migrator waits for a leader.
func awaitMigrationLeaderPoll(t *testing.T, ctx context.Context, observer *pgxpool.Pool, pid int32, done <-chan error) time.Time {
	return awaitMigrationLeaderPollAfter(t, ctx, observer, pid, time.Time{}, done)
}

// query_start is the PostgreSQL clock: a later observed poll proves the
// follower kept waiting beyond its DDL budget without a client-side sleep.
func awaitMigrationLeaderPollAfter(t *testing.T, ctx context.Context, observer *pgxpool.Pool, pid int32, after time.Time, done <-chan error) time.Time {
	t.Helper()
	tick := time.NewTicker(10 * time.Millisecond)
	defer tick.Stop()
	for {
		var query string
		var started time.Time
		var event *string
		if err := observer.QueryRow(ctx, `SELECT query, query_start, wait_event FROM pg_stat_activity WHERE pid = $1`, pid).
			Scan(&query, &started, &event); err != nil {
			t.Fatal(err)
		}
		if event != nil && *event == "advisory" {
			t.Fatal("migration follower used a blocking advisory wait instead of polling its leader")
		}
		if strings.Contains(query, "pg_try_advisory_xact_lock") && started.After(after) {
			return started
		}
		select {
		case err := <-done:
			t.Fatalf("migrator stopped before it waited for its leader: %v", err)
		case <-ctx.Done():
			t.Fatal(context.Cause(ctx))
		case <-tick.C:
		}
	}
}

func TestPostgresMigrationLeaderWaitOutlivesDDLLockTimeout(t *testing.T) {
	ctx := t.Context()
	leader, _ := isolatedPools(t, ctx, budgetTestDatabaseURL(t))
	follower := budgetTestPool(t, ctx, leader, PoolOptions{MaxConnections: 1, DisableWarmConnections: true})
	if _, err := ApplyMigrations(ctx, leader, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	locker, err := leader.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer locker.Rollback(ctx)
	if _, err := locker.Exec(ctx, `SELECT pg_advisory_xact_lock(`+migrationLockKeySQL+`)`, migrationLockKey); err != nil {
		t.Fatal(err)
	}
	connection, err := follower.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	pid := int32(connection.Conn().PgConn().PID())
	connection.Release()
	ddl := MigrationBudgets{StatementTimeout: time.Minute, LockTimeout: 100 * time.Millisecond}
	var result MigrationResult
	done := make(chan error, 1)
	go func() {
		var err error
		result, err = ApplyMigrations(ctx, follower, ddl)
		done <- err
	}()
	first := awaitMigrationLeaderPoll(t, ctx, leader, pid, done)
	awaitMigrationLeaderPollAfter(t, ctx, leader, pid, first.Add(ddl.LockTimeout), done)
	if err := locker.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err != nil || len(result.AppliedVersions) != 0 {
			t.Fatalf("follower after leader release: %+v %v", result, err)
		}
	case <-ctx.Done():
		t.Fatal("follower did not acquire released migration lock", ctx.Err())
	}
	migrationCtx, stop, err := WithMigrationBudgets(ctx, MigrationBudgets{
		StatementTimeout: 3 * time.Second, LockTimeout: 100 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	err = InTx(migrationCtx, follower, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := waitForMigrationLock(migrationCtx, tx); err != nil {
			return err
		}
		var statement, lock int64
		if err := tx.QueryRow(migrationCtx, `
SELECT (extract(epoch FROM current_setting('statement_timeout')::interval)*1000)::bigint,
       (extract(epoch FROM current_setting('lock_timeout')::interval)*1000)::bigint`).Scan(&statement, &lock); err != nil {
			return err
		}
		if statement != 3000 || lock != 100 {
			t.Errorf("DDL budgets after migration lock = %d/%d ms", statement, lock)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestPostgresOpenPoolDefaultsOverrideUnboundedServerSettings(t *testing.T) {
	ctx := t.Context()
	base, _ := isolatedPools(t, ctx, budgetTestDatabaseURL(t))
	pool := budgetTestPool(t, ctx, base, PoolOptions{})
	var statement, lock, idle string
	if err := pool.QueryRow(ctx, `SELECT current_setting('statement_timeout'),current_setting('lock_timeout'),current_setting('idle_in_transaction_session_timeout')`).Scan(&statement, &lock, &idle); err != nil {
		t.Fatal(err)
	}
	if statement != "15s" || lock != "2s" || idle != "30s" {
		t.Fatalf("unbounded/incorrect defaults: %s/%s/%s", statement, lock, idle)
	}
}
