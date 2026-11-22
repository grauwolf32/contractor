//go:build e2e

package uistack

import (
	"context"
	"os"
	"testing"

	"github.com/jackc/pgx/v5"
)

func TestUIProcessDatabaseIsolation(t *testing.T) {
	base := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if base == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	admin, err := pgx.Connect(t.Context(), base)
	if err != nil {
		t.Fatal(err)
	}
	defer admin.Close(context.Background())
	var parent, child string
	if err := admin.QueryRow(t.Context(), "SELECT current_database()").Scan(&parent); err != nil {
		t.Fatal(err)
	}
	t.Run("owned", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		conn, err := pgx.Connect(ctx, isolatedDatabase(t, ctx, base))
		if err != nil {
			t.Fatal(err)
		}
		defer conn.Close(context.Background())
		if err := conn.QueryRow(ctx, "SELECT current_database()").Scan(&child); err != nil {
			t.Fatal(err)
		}
		if child == parent {
			t.Fatal("UI Server shares the parent database-wide lease")
		}
	})
	var exists bool
	if err := admin.QueryRow(t.Context(), "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname=$1)", child).Scan(&exists); err != nil || exists {
		t.Fatalf("owned database cleanup: exists=%t error=%v", exists, err)
	}
}
