package artifacts_test

import (
	"context"
	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/jackc/pgx/v5/pgxpool"
	"os"
	"testing"
)

func testBlobContext(t *testing.T, ctx context.Context, pool *pgxpool.Pool) context.Context {
	t.Helper()
	if os.Getenv("CONTRACTOR_TEST_ARTIFACT_BACKEND") != "filesystem" {
		return ctx
	}
	files, err := OpenFilesystemBlobStore(ctx, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = files.Close() })
	return WithBlobRuntime(ctx, NewBlobRuntime(files, nil))
}
