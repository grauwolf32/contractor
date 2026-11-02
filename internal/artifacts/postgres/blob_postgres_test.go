package postgres_test

import (
	"context"
	"os"
	"testing"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	"github.com/jackc/pgx/v5/pgxpool"
)

func testBlobStorage(t *testing.T, ctx context.Context, pool *pgxpool.Pool) *artifactpostgres.Storage {
	t.Helper()
	if os.Getenv("CONTRACTOR_TEST_ARTIFACT_BACKEND") != "filesystem" {
		return artifactpgtest.NewStorage()
	}
	files, err := filesystem.Open(ctx, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = files.Close() })
	return artifactpgtest.NewConfigured(blob.NewRuntime(files, nil), artifactpostgres.DefaultHistoryPolicy())
}
