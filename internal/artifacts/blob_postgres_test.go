package artifacts_test

import (
	"context"
	"errors"
	. "github.com/grauwolf32/contractor/internal/artifacts"
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
