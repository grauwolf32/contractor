package postgrestest

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPoolIsolationConfigurationAndCancelledCleanup(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	inspector, err := pgx.Connect(t.Context(), databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer inspector.Close(context.Background())
	var schemas []string
	tracer := &queryCounter{}
	if !t.Run("isolated", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		for index := range 2 {
			inputURL := ""
			if index == 1 {
				inputURL = databaseURL
				t.Setenv("CONTRACTOR_TEST_DATABASE_URL", "")
			}
			pool := NewPool(t, ctx, inputURL, Options{Configure: func(config *pgxpool.Config) {
				config.MaxConns = 1
				config.ConnConfig.RuntimeParams["statement_timeout"] = "7000"
				config.ConnConfig.RuntimeParams["search_path"] = "public"
				config.ConnConfig.Tracer = tracer
			}})
			var schema, timeout string
			var empty bool
			if err := pool.QueryRow(ctx, "SELECT current_schema(), current_setting('statement_timeout'), to_regclass('workflow_runs') IS NULL").Scan(&schema, &timeout, &empty); err != nil {
				t.Fatal(err)
			}
			if schema == "public" || timeout != "7s" || pool.Config().MaxConns != 1 || !empty {
				t.Fatalf("isolation/configuration: schema=%s timeout=%s maxConns=%d empty=%t", schema, timeout, pool.Config().MaxConns, empty)
			}
			schemas = append(schemas, schema)
			if _, err := pool.Exec(ctx, "CREATE TABLE samples (value integer NOT NULL)"); err != nil {
				t.Fatal(err)
			}
			if _, err := pool.Exec(ctx, "INSERT INTO samples VALUES ($1)", index); err != nil {
				t.Fatal(err)
			}
			var count, value int
			if err := pool.QueryRow(ctx, "SELECT count(*), min(value) FROM samples").Scan(&count, &value); err != nil || count != 1 || value != index {
				t.Fatalf("isolated rows: count=%d value=%d err=%v", count, value, err)
			}
		}
	}) {
		return
	}
	if schemas[0] == schemas[1] || tracer.calls.Load() == 0 {
		t.Fatal("schema identities or caller tracer were not preserved")
	}
	for _, schema := range schemas {
		var exists bool
		if err := inspector.QueryRow(t.Context(), "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname=$1)", schema).Scan(&exists); err != nil || exists {
			t.Fatalf("schema survived cancelled cleanup: exists=%t err=%v", exists, err)
		}
	}
}

func TestMissingDatabaseSkipsAndConfiguredFailureFails(t *testing.T) {
	for _, scenario := range []string{"unset", "unreachable", "invalid"} {
		t.Run(scenario, func(t *testing.T) {
			output, err := poolSubprocess(t, scenario, "")
			if scenario == "unset" {
				if err != nil || !strings.Contains(output, "--- SKIP:") {
					t.Fatalf("missing URL did not skip: %s", output)
				}
			} else if err == nil || strings.Contains(output, "--- SKIP:") {
				t.Fatalf("configured failure did not fail: %s", output)
			}
		})
	}
}

func TestSetupFailureDropsCreatedSchema(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	path := filepath.Join(t.TempDir(), "schema")
	output, err := poolSubprocess(t, "setup-failure", path)
	if err == nil || !strings.Contains(output, "injected setup failure") {
		t.Fatalf("setup did not reach the injected failure: %s", output)
	}
	schema, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	inspector, err := pgx.Connect(t.Context(), databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer inspector.Close(context.Background())
	var exists bool
	if err := inspector.QueryRow(t.Context(), "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname=$1)", string(schema)).Scan(&exists); err != nil || exists {
		t.Fatalf("schema survived setup failure: exists=%t err=%v", exists, err)
	}
}

// The subprocess proves testing.T fatal/skip and Cleanup behavior on the real
// helper path without marking the parent test failed or replacing testing.TB.
func TestPoolSubprocess(t *testing.T) {
	scenario := os.Getenv("CONTRACTOR_POOL_TEST_SCENARIO")
	options := Options{}
	databaseURL := ""
	switch scenario {
	case "":
		return
	case "unset":
		t.Setenv("CONTRACTOR_TEST_DATABASE_URL", "")
	case "unreachable":
		t.Setenv("CONTRACTOR_TEST_DATABASE_URL", "postgres://127.0.0.1:1/unavailable?sslmode=disable&connect_timeout=1")
	case "invalid":
		t.Setenv("CONTRACTOR_TEST_DATABASE_URL", "postgres://%")
	case "setup-failure":
		databaseURL = os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
		// Explicit URLs must work even when the environment has no URL.
		t.Setenv("CONTRACTOR_TEST_DATABASE_URL", "")
		options.Configure = func(config *pgxpool.Config) {
			if err := os.WriteFile(os.Getenv("CONTRACTOR_POOL_TEST_SCHEMA"), []byte(config.ConnConfig.RuntimeParams["search_path"]), 0o600); err != nil {
				t.Fatal(err)
			}
			t.Fatal("injected setup failure")
		}
	default:
		t.Fatal("unexpected helper scenario")
	}
	NewPool(t, t.Context(), databaseURL, options)
	t.Fatal("expected helper to stop the subprocess test")
}

func poolSubprocess(t *testing.T, scenario, path string) (string, error) {
	t.Helper()
	command := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestPoolSubprocess$", "-test.v")
	command.Env = append(os.Environ(), "CONTRACTOR_POOL_TEST_SCENARIO="+scenario, "CONTRACTOR_POOL_TEST_SCHEMA="+path)
	output, err := command.CombinedOutput()
	return string(output), err
}

type queryCounter struct{ calls atomic.Int64 }

func (q *queryCounter) TraceQueryStart(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryStartData) context.Context {
	q.calls.Add(1)
	return ctx
}

func (*queryCounter) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}
