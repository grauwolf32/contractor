package postgres

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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

func TestRuntimeConfigurationMigrationContainsExactIdempotentBootstrap(t *testing.T) {
	t.Parallel()

	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	latest := items[len(items)-1]
	if latest.name != "000015_runtime_configuration.sql" {
		t.Fatalf("latest migration = %q, want RuntimeConfig migration", latest.name)
	}
	canonical := `{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contractor-empty","version":"1"},"spec":{}}`
	sum := sha256.Sum256([]byte(canonical))
	digest := "sha256:" + hex.EncodeToString(sum[:])
	if digest != "sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f" {
		t.Fatalf("test fixture digest = %s", digest)
	}
	contents := string(latest.contents)
	for _, required := range []string{
		canonical,
		digest,
		"'default'",
		"ON CONFLICT (name, version) DO NOTHING",
		"ON CONFLICT (label) DO NOTHING",
		"revision numeric(20, 0)",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("RuntimeConfig migration does not contain %q", required)
		}
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
