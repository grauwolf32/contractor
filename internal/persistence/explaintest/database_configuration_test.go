package explaintest

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

// A subprocess observes the real testing.T skip/fatal behavior. A deliberately
// invalid ambient PGHOST cannot replace the explicit test database setting.
func TestDatabaseRequiresConfiguredURL(t *testing.T) {
	for _, scenario := range []string{"unset", "invalid", "unavailable", "explicit"} {
		t.Run(scenario, func(t *testing.T) {
			command := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestDatabaseConfigurationSubprocess$", "-test.v")
			command.Env = append(os.Environ(), "CONTRACTOR_EXPLAIN_DATABASE_SCENARIO="+scenario)
			output, err := command.CombinedOutput()
			if scenario == "unset" {
				if err != nil || !strings.Contains(string(output), "--- SKIP:") || !strings.Contains(string(output), "CONTRACTOR_TEST_DATABASE_URL is not set") {
					t.Fatalf("unset test database did not skip before connecting: %s", output)
				}
				return
			}
			if err == nil || strings.Contains(string(output), "--- SKIP:") {
				t.Fatalf("configured database failure did not fail: %s", output)
			}
			want := "configured PostgreSQL test database is unavailable"
			if scenario == "invalid" {
				want = "invalid PostgreSQL test configuration"
			}
			if !strings.Contains(string(output), want) {
				t.Fatalf("wrong configuration path, want %q: %s", want, output)
			}
		})
	}
}

func TestDatabaseConfigurationSubprocess(t *testing.T) {
	scenario := os.Getenv("CONTRACTOR_EXPLAIN_DATABASE_SCENARIO")
	if scenario == "" {
		return
	}
	t.Setenv("PGHOST", "/nonexistent-contractor-review-pg-socket")
	argument := ""
	switch scenario {
	case "unset":
		t.Setenv("CONTRACTOR_TEST_DATABASE_URL", "")
	case "invalid":
		t.Setenv("CONTRACTOR_TEST_DATABASE_URL", "postgres://%")
	case "unavailable":
		t.Setenv("CONTRACTOR_TEST_DATABASE_URL", "postgres://127.0.0.1:1/unavailable?sslmode=disable&connect_timeout=1")
	case "explicit":
		// The valid explicit URL takes precedence over malformed environment.
		t.Setenv("CONTRACTOR_TEST_DATABASE_URL", "postgres://%")
		argument = "postgres://127.0.0.1:1/unavailable?sslmode=disable&connect_timeout=1"
	default:
		t.Fatal("unknown test database scenario")
	}
	Database(t, t.Context(), argument, func(context.Context, *pgxpool.Pool) error {
		t.Fatal("unexpected database setup")
		return nil
	})
	t.Fatal("expected helper to stop the subprocess test")
}
