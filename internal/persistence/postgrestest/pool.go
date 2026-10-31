// Package postgrestest opens schema-isolated PostgreSQL pools for real-store
// tests. Callers own migrations, fixtures and operation-specific settings.
package postgrestest

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Options preserves the caller's connection settings and cleanup budget.
type Options struct {
	Configure func(*pgxpool.Config)
	// CleanupTimeout defaults to ten seconds.
	CleanupTimeout time.Duration
}

// NewPool creates an empty, isolated schema and a verified pool using it.
// An empty databaseURL uses CONTRACTOR_TEST_DATABASE_URL, skipping only when
// that variable is unset. A configured but unavailable database fails the test.
// Configure may adjust the work pool; search_path always names the new schema.
// Cleanup closes the pool, drops the schema with an independent bounded context
// and closes the admin pool, including when setup fails. Migrations are explicit
// at the call site so migration tests can also start with an empty schema.
func NewPool(t testing.TB, ctx context.Context, databaseURL string, options Options) *pgxpool.Pool {
	t.Helper()
	if databaseURL == "" {
		databaseURL = os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
		if databaseURL == "" {
			t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
		}
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal("invalid PostgreSQL test configuration")
	}
	admin, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal("open PostgreSQL test admin pool:", err)
	}
	t.Cleanup(admin.Close)
	if err := admin.Ping(ctx); err != nil {
		t.Fatal("configured PostgreSQL test database is unavailable:", err)
	}
	var suffix [8]byte
	if _, err := rand.Read(suffix[:]); err != nil {
		t.Fatal("create PostgreSQL test schema identity:", err)
	}
	schema := "contractor_test_" + hex.EncodeToString(suffix[:])
	identifier := pgx.Identifier{schema}.Sanitize()
	// Register before CREATE: a lost acknowledgement may still have created it.
	t.Cleanup(func() {
		budget := options.CleanupTimeout
		if budget == 0 {
			budget = 10 * time.Second
		}
		cleanup, cancel := context.WithTimeout(context.Background(), budget)
		defer cancel()
		if _, err := admin.Exec(cleanup, "DROP SCHEMA IF EXISTS "+identifier+" CASCADE"); err != nil {
			t.Error("drop PostgreSQL test schema:", err)
		}
	})
	if _, err := admin.Exec(ctx, "CREATE SCHEMA "+identifier); err != nil {
		t.Fatal("create PostgreSQL test schema:", err)
	}
	config = config.Copy()
	config.ConnConfig.RuntimeParams["search_path"] = schema
	if options.Configure != nil {
		options.Configure(config)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal("open isolated PostgreSQL test pool:", err)
	}
	t.Cleanup(pool.Close)
	if err := pool.Ping(ctx); err != nil {
		t.Fatal("connect isolated PostgreSQL test pool:", err)
	}
	return pool
}
