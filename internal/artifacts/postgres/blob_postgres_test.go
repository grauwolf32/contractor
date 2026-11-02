package postgres_test

import (
	"context"
	"os"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	"github.com/jackc/pgx/v5/pgxpool"
)

func testBlobContext(t *testing.T, ctx context.Context, pool *pgxpool.Pool) context.Context {
	t.Helper()
	if os.Getenv("CONTRACTOR_TEST_ARTIFACT_BACKEND") != "filesystem" {
		return ctx
	}
	files, err := filesystem.Open(ctx, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = files.Close() })
	return blob.WithRuntime(ctx, blob.NewRuntime(files, nil))
}
