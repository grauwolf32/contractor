package postgres

import (
	"context"
	"crypto/sha256"
	"errors"
	"strings"
	"testing"
)

func TestEmbeddedMigrationsAreOrderedAndExcludeRuntimeLiveness(t *testing.T) {
	t.Parallel()

	items, err := loadMigrations()
	if err != nil {
		t.Fatalf("loadMigrations: %v", err)
	}
	if len(items) == 0 {
		t.Fatal("no migrations loaded")
	}
	previous := int64(0)
	for _, item := range items {
		if item.version <= previous {
			t.Fatalf("migration versions are not increasing: %d after %d", item.version, previous)
		}
		previous = item.version
		lower := strings.ToLower(string(item.contents))
		for _, forbidden := range []string{"runtime_agents", "heartbeat_lease", "create type"} {
			if strings.Contains(lower, forbidden) {
				t.Fatalf("migration %s contains forbidden durable construct %q", item.name, forbidden)
			}
		}
	}
}

func TestCompareMigrationLedgerClassifiesPendingAndDrift(t *testing.T) {
	first := migration{version: 1, name: "000001_first.sql", checksum: sha256.Sum256([]byte("first"))}
	second := migration{version: 2, name: "000002_second.sql", checksum: sha256.Sum256([]byte("second"))}
	available := []migration{first, second}
	validFirst := appliedMigration{name: first.name, checksum: first.checksum}
	validSecond := appliedMigration{name: second.name, checksum: second.checksum}
	for _, test := range []struct {
		name         string
		applied      map[int64]appliedMigration
		current      int64
		pending      int
		migrationErr error
	}{
		{name: "empty", applied: map[int64]appliedMigration{}, pending: 2},
		{name: "pending", applied: map[int64]appliedMigration{1: validFirst}, current: 1, pending: 1},
		{name: "exact", applied: map[int64]appliedMigration{1: validFirst, 2: validSecond}, current: 2},
		{name: "unknown", applied: map[int64]appliedMigration{1: validFirst, 3: {name: "000003_future.sql"}}, migrationErr: ErrMigrationDrift},
		{name: "renamed", applied: map[int64]appliedMigration{1: {name: "renamed.sql", checksum: first.checksum}}, migrationErr: ErrMigrationDrift},
		{name: "checksum drift", applied: map[int64]appliedMigration{1: {name: first.name, checksum: second.checksum}}, migrationErr: ErrMigrationDrift},
	} {
		t.Run(test.name, func(t *testing.T) {
			state, err := compareMigrationLedger(available, test.applied)
			if !errors.Is(err, test.migrationErr) || (test.migrationErr == nil &&
				(state.currentVersion != test.current || len(state.pending) != test.pending)) {
				t.Fatalf("state=%+v error=%v, want current=%d pending=%d error=%v",
					state, err, test.current, test.pending, test.migrationErr)
			}
		})
	}
}

func TestInvalidDatabaseConfigurationIsRedacted(t *testing.T) {
	t.Parallel()

	const secret = "not-a-real-secret"
	pool, err := OpenPool(context.Background(), "://"+secret, PoolOptions{})
	if pool != nil {
		pool.Close()
		t.Fatal("OpenPool returned a pool for an invalid URL")
	}
	if !errors.Is(err, ErrInvalidDatabaseConfiguration) {
		t.Fatalf("OpenPool error = %v", err)
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("OpenPool error exposed database configuration: %v", err)
	}
}
