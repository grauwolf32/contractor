//go:build integration

package settingsstore

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"errors"
	"os"
	"sync"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresSchedulerSettingsSeedCASRestartAndFailClosed(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool, reopen := isolatedSettingsPool(t, ctx, databaseURL)
	store := NewPostgresStore(pool)

	initial, err := store.GetSchedulerSettings(ctx)
	if err != nil || initial.MaxConcurrentRuns != 1 || initial.Revision != 1 ||
		initial.UpdatedAt.IsZero() || initial.UpdatedAt.Location() != time.UTC {
		t.Fatalf("initial Scheduler settings = (%+v, %v)", initial, err)
	}
	unchanged, err := store.UpdateSchedulerSettings(ctx, UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 1, ExpectedRevision: initial.Revision,
	})
	if err != nil || unchanged != initial {
		t.Fatalf("no-op Scheduler settings replacement = (%+v, %v), want %+v", unchanged, err, initial)
	}

	for _, statement := range []string{
		`UPDATE scheduler_settings SET max_concurrent_runs = 0, revision = 2 WHERE singleton = true`,
		`UPDATE scheduler_settings SET revision = 18446744073709551616 WHERE singleton = true`,
		`DELETE FROM scheduler_settings WHERE singleton = true`,
	} {
		if _, err := pool.Exec(ctx, statement); persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation {
			t.Fatalf("invalid direct mutation %q: error = %v", statement, err)
		}
	}

	start := make(chan struct{})
	results := make([]SchedulerSettings, 2)
	errorsByCall := make([]error, 2)
	var wait sync.WaitGroup
	for index, value := range []int{2, 3} {
		wait.Add(1)
		go func(index, value int) {
			defer wait.Done()
			<-start
			results[index], errorsByCall[index] = store.UpdateSchedulerSettings(ctx, UpdateSchedulerSettingsParams{
				MaxConcurrentRuns: value, ExpectedRevision: initial.Revision,
			})
		}(index, value)
	}
	close(start)
	wait.Wait()
	winners := 0
	preconditions := 0
	var committed SchedulerSettings
	for index, callErr := range errorsByCall {
		if callErr == nil {
			winners++
			committed = results[index]
		} else if errors.Is(callErr, ErrPrecondition) {
			preconditions++
		} else {
			t.Fatalf("concurrent update %d: %v", index, callErr)
		}
	}
	if winners != 1 || preconditions != 1 || committed.Revision != 2 ||
		committed.UpdatedAt.Before(initial.UpdatedAt) || committed.UpdatedAt.Equal(initial.UpdatedAt) {
		t.Fatalf("concurrent results = %+v / %v", results, errorsByCall)
	}

	restartedPool := reopen(t, ctx)
	defer restartedPool.Close()
	restarted, err := NewPostgresStore(restartedPool).GetSchedulerSettings(ctx)
	if err != nil || restarted != committed {
		t.Fatalf("restarted Scheduler settings = (%+v, %v), want %+v", restarted, err, committed)
	}
	if _, err := store.UpdateSchedulerSettings(ctx, UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: committed.MaxConcurrentRuns, ExpectedRevision: initial.Revision,
	}); !errors.Is(err, ErrPrecondition) {
		t.Fatalf("stale Scheduler settings update error = %v", err)
	}

	if _, err := pool.Exec(ctx, `ALTER TABLE scheduler_settings DISABLE TRIGGER scheduler_settings_protect_mutation`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM scheduler_settings WHERE singleton = true`); err != nil {
		t.Fatal(err)
	}
	if _, err := store.GetSchedulerSettings(ctx); !errors.Is(err, ErrInvariant) {
		t.Fatalf("missing singleton read error = %v", err)
	}
	if err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		_, err := NewPostgresTxStore(tx).LockSchedulerSettings(ctx)
		return err
	}); !errors.Is(err, ErrInvariant) {
		t.Fatalf("missing singleton lock error = %v", err)
	}
}

// TestPostgresSchedulerSettingsLockOrdersAChange checks the order an Audit
// dispatch reservation relies on (AUX-R6): a change waits for the transaction
// that locked the settings row, and a transaction that locks the row after a
// change committed reads the changed value.
func TestPostgresSchedulerSettingsLockOrdersAChange(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedSettingsPool(t, ctx, databaseURL)

	locker, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = locker.Rollback(context.Background()) }()
	locked, err := NewPostgresTxStore(locker).LockSchedulerSettings(ctx)
	if err != nil || locked.MaxConcurrentRuns != 1 || locked.Revision != 1 {
		t.Fatalf("locked Scheduler settings = (%+v, %v)", locked, err)
	}

	updater, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer updater.Release()
	var updaterPID int
	if err := updater.QueryRow(ctx, `SELECT pg_backend_pid()`).Scan(&updaterPID); err != nil {
		t.Fatal(err)
	}
	type updateResult struct {
		settings SchedulerSettings
		err      error
	}
	updated := make(chan updateResult, 1)
	go func() {
		settings, err := NewPostgresStore(updater).UpdateSchedulerSettings(ctx, UpdateSchedulerSettingsParams{
			MaxConcurrentRuns: 4, ExpectedRevision: locked.Revision,
		})
		updated <- updateResult{settings: settings, err: err}
	}()
	for waiting := false; !waiting; {
		select {
		case result := <-updated:
			t.Fatalf("settings change committed while the row was locked: (%+v, %v)", result.settings, result.err)
		case <-ctx.Done():
			t.Fatal("settings change never waited for the locked row")
		case <-time.After(10 * time.Millisecond):
		}
		var waitEventType *string
		if err := pool.QueryRow(ctx, `
SELECT wait_event_type FROM pg_stat_activity WHERE pid = $1`, updaterPID).Scan(&waitEventType); err != nil {
			t.Fatal(err)
		}
		waiting = waitEventType != nil && *waitEventType == "Lock"
	}
	if err := locker.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	result := <-updated
	if result.err != nil || result.settings.MaxConcurrentRuns != 4 || result.settings.Revision != 2 {
		t.Fatalf("settings change after the lock = (%+v, %v)", result.settings, result.err)
	}

	if err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		relocked, err := NewPostgresTxStore(tx).LockSchedulerSettings(ctx)
		if err == nil && relocked != result.settings {
			t.Errorf("lock after the change = %+v, want %+v", relocked, result.settings)
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
}

func isolatedSettingsPool(
	t *testing.T,
	ctx context.Context,
	databaseURL string,
) (*pgxpool.Pool, func(*testing.T, context.Context) *pgxpool.Pool) {
	t.Helper()
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatalf("CONTRACTOR_TEST_DATABASE_URL is set but PostgreSQL is unreachable: %v", err)
	}
	suffix := make([]byte, 8)
	if _, err := cryptorand.Read(suffix); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_settings_" + hex.EncodeToString(suffix)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	open := func(t *testing.T, ctx context.Context) *pgxpool.Pool {
		t.Helper()
		config := adminConfig.Copy()
		config.ConnConfig.RuntimeParams["search_path"] = schema
		pool, err := pgxpool.NewWithConfig(ctx, config)
		if err != nil {
			t.Fatal(err)
		}
		if err := pool.Ping(ctx); err != nil {
			pool.Close()
			t.Fatal(err)
		}
		return pool
	}
	pool := open(t, ctx)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`)
		admin.Close()
	})
	return pool, open
}
