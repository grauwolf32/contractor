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

func TestPostgresOrdinaryWaitBudgetsAndRecovery(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	base, _ := isolatedPools(t, ctx, budgetTestDatabaseURL(t))
	var logs bytes.Buffer
	pool := budgetTestPool(t, ctx, base, PoolOptions{MaxConnections: 1, Budgets: smallDatabaseBudgets(), Logger: slog.New(slog.NewJSONHandler(&logs, nil))})
	if _, err := pool.Exec(ctx, `CREATE TABLE budget_rows (id integer PRIMARY KEY,value integer NOT NULL); INSERT INTO budget_rows VALUES(1,0)`); err != nil {
		t.Fatal(err)
	}
	assertReusable := func() {
		t.Helper()
		var value int
		if err := pool.QueryRow(ctx, `SELECT value FROM budget_rows WHERE id=1`).Scan(&value); err != nil || value != 0 {
			t.Fatalf("pool/rollback not reusable: value=%d error=%v", value, err)
		}
	}
	assertBounded := func(start time.Time) {
		t.Helper()
		if time.Since(start) > 2*time.Second {
			t.Fatalf("operation escaped its budget: %s", time.Since(start))
		}
	}
	t.Run("exhausted-pool", func(t *testing.T) {
		held, err := pool.Acquire(ctx)
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		_, err = pool.Exec(context.Background(), `SELECT 'secret-sql-argument'`)
		assertBounded(start)
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
		assertReusable()
	})
	t.Run("statement-abort-no-retry", func(t *testing.T) {
		calls := 0
		start := time.Now()
		err := InTxWithRetry(context.Background(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			calls++
			if _, err := tx.Exec(context.Background(), `UPDATE budget_rows SET value=1`); err != nil {
				return err
			}
			_, err := tx.Exec(context.Background(), `SELECT pg_sleep(3)`)
			return err
		})
		assertBounded(start)
		if SQLState(err) != "57014" || calls != 1 || IsTransactionConflict(err) {
			t.Fatalf("statement timeout: state=%s attempts=%d error=%v", SQLState(err), calls, err)
		}
		assertReusable()
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
		start := time.Now()
		calls := 0
		err = InTxWithRetry(context.Background(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			calls++
			_, err := tx.Exec(context.Background(), `UPDATE budget_rows SET value=2 WHERE id=1`)
			return err
		})
		assertBounded(start)
		if SQLState(err) != "55P03" || calls != 1 || IsTransactionConflict(err) {
			t.Fatalf("lock timeout: state=%s attempts=%d error=%v", SQLState(err), calls, err)
		}
		if err := locker.Rollback(ctx); err != nil {
			t.Fatal(err)
		}
		assertReusable()
	})
	t.Run("client-budget-even-with-server-timeout-disabled", func(t *testing.T) {
		start := time.Now()
		err := InTx(context.Background(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			if _, err := tx.Exec(context.Background(), `SET LOCAL statement_timeout=0`); err != nil {
				return err
			}
			if _, err := tx.Exec(context.Background(), `UPDATE budget_rows SET value=3`); err != nil {
				return err
			}
			_, err := tx.Exec(context.Background(), `SELECT pg_sleep(3)`)
			return err
		})
		assertBounded(start)
		if !errors.Is(err, context.DeadlineExceeded) || IsTransactionConflict(err) {
			t.Fatalf("client timeout identity: %v", err)
		}
		assertReusable()
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
		assertReusable()
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
		deadline := time.NewTimer(2 * time.Second)
		defer deadline.Stop()
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
			select {
			case <-deadline.C:
				t.Fatal("idle transaction was not reaped")
			case <-tick.C:
			}
		}
		_ = tx.Rollback(ctx)
		assertReusable()
	})
}

func TestPostgresMaintenanceBudgetsRestorePoolSettings(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	base, _ := isolatedPools(t, ctx, budgetTestDatabaseURL(t))
	pool := budgetTestPool(t, ctx, base, PoolOptions{MaxConnections: 1, Budgets: smallDatabaseBudgets()})
	check := func(ctx context.Context, db DBTX, want Budgets) {
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
	check(ctx, pool, smallDatabaseBudgets())
	for _, budget := range []func(context.Context) (context.Context, context.CancelFunc){WithMigrationBudget, WithCleanupBudget} {
		for _, rollback := range []bool{false, true} {
			maintenance, done := budget(ctx)
			err := InTx(maintenance, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				check(maintenance, tx, maintenance.Value(maintenanceBudgetKey{}).(Budgets))
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
			check(ctx, pool, smallDatabaseBudgets())
		}
	}
	if _, err := ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	if result, err := ApplyMigrations(ctx, pool); err != nil || len(result.AppliedVersions) != 0 {
		t.Fatalf("migration replay: %+v %v", result, err)
	}
	check(ctx, pool, smallDatabaseBudgets())
}

func TestPostgresMigrationLockCancellationLeavesCapacityReusable(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	base, _ := isolatedPools(t, ctx, budgetTestDatabaseURL(t))
	pool := budgetTestPool(t, ctx, base, PoolOptions{MaxConnections: 1})
	locker, err := base.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer locker.Rollback(ctx)
	if _, err := locker.Exec(ctx, `SELECT pg_advisory_xact_lock($1)`, migrationLockKey); err != nil {
		t.Fatal(err)
	}
	short, stop := context.WithTimeout(ctx, 100*time.Millisecond)
	_, err = ApplyMigrations(short, pool)
	stop()
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("migration lock cancellation: %v", err)
	}
	if err := locker.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool); err != nil {
		t.Fatalf("migration after cancellation: %v", err)
	}
}

func TestPostgresOpenPoolDefaultsOverrideUnboundedServerSettings(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
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
