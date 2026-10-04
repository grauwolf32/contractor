package postgres

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// A leader holding one schema's migration lock must not stall migrators of
// another schema in the same database, while a second migrator of the held
// schema still waits for its leader.
func TestPostgresMigrationLeaderIsScopedToTargetSchema(t *testing.T) {
	databaseURL := budgetTestDatabaseURL(t)
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	held, heldFollower := isolatedPools(t, ctx, databaseURL)
	other, otherFollower := isolatedPools(t, ctx, databaseURL)

	// Hold exactly the lock a migration leader of the first schema holds.
	locker, err := held.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer locker.Rollback(ctx)
	if err := waitForMigrationLock(ctx, locker); err != nil {
		t.Fatal(err)
	}

	// Both migrators of the other schema race; one applies every version.
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	type outcome struct {
		result MigrationResult
		err    error
	}
	otherCtx, stopOther := context.WithTimeout(ctx, time.Minute)
	defer stopOther()
	done := make(chan outcome, 2)
	for _, pool := range []*pgxpool.Pool{other, otherFollower} {
		go func(pool *pgxpool.Pool) {
			result, err := ApplyMigrations(otherCtx, pool)
			done <- outcome{result, err}
		}(pool)
	}
	applied := 0
	for range 2 {
		finished := <-done
		if finished.err != nil {
			t.Fatalf("migration of an independent schema waited for another schema's leader: %v", finished.err)
		}
		applied += len(finished.result.AppliedVersions)
	}
	if applied != len(available) {
		t.Fatalf("independent schema applied %d versions across both migrators, want %d once", applied, len(available))
	}

	short, stop := context.WithTimeout(ctx, 500*time.Millisecond)
	_, err = ApplyMigrations(short, heldFollower)
	stop()
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("second migrator of the held schema = %v, want to wait for its leader", err)
	}
	if err := locker.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(ctx, heldFollower)
	if err != nil || len(result.AppliedVersions) != len(available) {
		t.Fatalf("migration after the leader released its schema: %d versions, %v", len(result.AppliedVersions), err)
	}
}

func TestPostgresMigrationRequiresAnExistingTargetSchema(t *testing.T) {
	databaseURL := budgetTestDatabaseURL(t)
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = "contractor_missing_" + randomHex(t, 8)
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	_, err = ApplyMigrations(ctx, pool)
	if err == nil || !strings.Contains(err.Error(), "no existing schema") || errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("migration without a target schema = %v, want an immediate schema error", err)
	}
}
