package postgrestest

import (
	"context"
	"net/url"
	"os"
	"testing"

	"github.com/jackc/pgx/v5"
)

func TestDatabaseIsolationOptionsAndCancelledCleanup(t *testing.T) {
	base := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if base == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	admin, err := pgx.Connect(t.Context(), base)
	if err != nil {
		t.Fatal(err)
	}
	defer admin.Close(context.Background())
	var parent string
	if err := admin.QueryRow(t.Context(), "SELECT current_database()").Scan(&parent); err != nil {
		t.Fatal(err)
	}
	parsed, err := url.Parse(base)
	if err != nil {
		t.Fatal("invalid test URL")
	}
	query := parsed.Query()
	query.Set("search_path", "not_the_process_schema")
	query.Set("application_name", "process-database-regression")
	parsed.RawQuery = query.Encode()
	var names []string
	if !t.Run("owned", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		for range 2 {
			dsn := NewDatabase(t, ctx, parsed.String())
			actual, err := url.Parse(dsn)
			if err != nil || actual.Query().Has("search_path") || actual.Query().Get("application_name") != "process-database-regression" {
				t.Fatal("process database options were not isolated")
			}
			conn, err := pgx.Connect(ctx, dsn)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := conn.Close(context.Background()); err != nil {
					t.Error(err)
				}
			})
			var name, application string
			if err := conn.QueryRow(ctx, "SELECT current_database(),current_setting('application_name')").Scan(&name, &application); err != nil {
				t.Fatal(err)
			}
			if name == parent || application != "process-database-regression" {
				t.Fatal("parent database or connection options leaked")
			}
			names = append(names, name)
		}
	}) {
		return
	}
	if len(names) != 2 || names[0] == names[1] {
		t.Fatal("process databases share a lease")
	}
	for _, name := range names {
		var exists bool
		if err := admin.QueryRow(t.Context(), "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname=$1)", name).Scan(&exists); err != nil || exists {
			t.Fatalf("owned database survived cancelled-context cleanup: exists=%t error=%v", exists, err)
		}
	}
}
