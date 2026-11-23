package postgrestest

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// NewDatabase isolates a process Server's database-wide lease. Repository-only
// tests use NewPool's cheaper schema isolation instead. The caller registers
// process/pool cleanup after this call, so they finish before database deletion.
// Cleanup is registered before CREATE, covering an ambiguous acknowledgement,
// and uses an independent bounded context even if setup was cancelled.
func NewDatabase(t testing.TB, ctx context.Context, databaseURL string) string {
	t.Helper()
	if databaseURL == "" {
		databaseURL = os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
		if databaseURL == "" {
			t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
		}
	}
	parsed, err := url.Parse(databaseURL)
	if err != nil || parsed.Scheme != "postgres" && parsed.Scheme != "postgresql" {
		t.Fatal("PostgreSQL process test configuration must be a PostgreSQL URL")
	}
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal("open PostgreSQL process test admin pool")
	}
	t.Cleanup(admin.Close)
	if err := admin.Ping(ctx); err != nil {
		t.Fatal("configured PostgreSQL process test database is unavailable")
	}
	var suffix [8]byte
	if _, err := rand.Read(suffix[:]); err != nil {
		t.Fatal("create PostgreSQL process test identity")
	}
	name := "contractor_process_" + hex.EncodeToString(suffix[:])
	identifier := pgx.Identifier{name}.Sanitize()
	cleanupPossible := true
	t.Cleanup(func() {
		if !cleanupPossible {
			return
		}
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, "DROP DATABASE IF EXISTS "+identifier+" WITH (FORCE)"); err != nil {
			t.Error("drop owned PostgreSQL process test database:", err)
		}
	})
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+identifier); err != nil {
		// A PostgreSQL rejection is definite (including a name collision).
		// Only a lost response may have created our database and needs cleanup.
		var rejected *pgconn.PgError
		cleanupPossible = !errors.As(err, &rejected)
		t.Fatal("create owned PostgreSQL process test database:", err)
	}
	parsed.Path, parsed.RawPath = "/"+name, ""
	query := parsed.Query()
	query.Del("search_path")
	parsed.RawQuery = query.Encode()
	return parsed.String()
}
