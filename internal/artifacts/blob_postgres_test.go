package artifacts_test

import (
	"context"
	"errors"
	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/jackc/pgx/v5/pgxpool"
	"os"
	"testing"
	"time"
)

func TestPostgresBlobBackendClaimAndGuard(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	if err := ClaimBlobBackend(ctx, pool, BlobFilesystem); err != nil {
		t.Fatal(err)
	}
	if err := ClaimBlobBackend(ctx, pool, BlobFilesystem); err != nil {
		t.Fatal(err)
	}
	if err := ClaimBlobBackend(ctx, pool, BlobPostgres); !errors.Is(err, ErrBlobBackendMismatch) {
		t.Fatalf("mismatch: %v", err)
	}
	_, err := pool.Exec(ctx, `INSERT INTO artifact_blobs(sha256,payload,size_bytes) VALUES (sha256('x'::bytea),'x'::bytea,1)`)
	if err == nil {
		t.Fatal("default inline writer bypassed filesystem selection")
	}
}

func testBlobContext(t *testing.T, ctx context.Context, pool *pgxpool.Pool) context.Context {
	t.Helper()
	if os.Getenv("CONTRACTOR_TEST_ARTIFACT_BACKEND") != "filesystem" {
		return ctx
	}
	if err := ClaimBlobBackend(ctx, pool, BlobFilesystem); err != nil {
		t.Fatal(err)
	}
	files, err := OpenFilesystemBlobStore(ctx, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = files.Close() })
	return WithBlobRuntime(ctx, NewBlobRuntime(files, nil))
}
