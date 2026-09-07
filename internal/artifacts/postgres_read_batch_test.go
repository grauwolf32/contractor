package artifacts_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
)

func TestReadExactBatchPreservesScopeRevisionAndBlobIntegrity(t *testing.T) {
	for _, backend := range []BlobBackend{BlobPostgres, BlobFilesystem} {
		t.Run(string(backend), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			pool := isolatedArtifactPool(t, ctx)
			if err := ClaimBlobBackend(ctx, pool, backend); err != nil {
				t.Fatal(err)
			}
			var blobPath string
			if backend == BlobFilesystem {
				blobPath = t.TempDir()
				files, err := OpenFilesystemBlobStore(ctx, blobPath)
				if err != nil {
					t.Fatal(err)
				}
				defer files.Close()
				ctx = WithBlobRuntime(ctx, NewBlobRuntime(files, nil))
			}
			repository := NewPostgresRepository(pool)
			scope, _ := UserScope("batch-owner")
			foreignScope, _ := UserScope("another-owner")
			logical := ArtifactRef{Namespace: "documents", Name: "proposal"}
			first, err := repository.Write(ctx, scope, logical, Payload{MediaType: "application/json", Data: []byte(`{"version":1}`)}, nil)
			if err != nil {
				t.Fatal(err)
			}
			latest, err := repository.Write(ctx, scope, logical, Payload{MediaType: "application/json", Data: []byte(`{"version":2}`)}, first.Ref.Revision)
			if err != nil {
				t.Fatal(err)
			}
			foreign, err := repository.Write(ctx, foreignScope, logical, Payload{MediaType: "text/plain", Data: []byte("foreign")}, nil)
			if err != nil {
				t.Fatal(err)
			}
			requests := []ExactReadRequest{{Scope: scope, Ref: latest.Ref}, {Scope: scope, Ref: first.Ref}, {Scope: foreignScope, Ref: foreign.Ref}, {Scope: scope, Ref: first.Ref}}
			reads, err := repository.ReadExactBatch(ctx, requests)
			if err != nil || len(reads) != len(requests) {
				t.Fatalf("batch=%v error=%v", reads, err)
			}
			for index, request := range requests {
				single, err := repository.Read(ctx, request.Scope, request.Ref)
				if err != nil || !reflect.DeepEqual(single, reads[index]) {
					t.Fatalf("batch result %d differs: %v", index, err)
				}
			}
			requests[2].Scope = scope
			if result, err := repository.ReadExactBatch(ctx, requests); result != nil || !errors.Is(err, ErrArtifactNotFound) {
				t.Fatalf("foreign revision: %v, %v", result, err)
			}
			if _, err := repository.ReadExactBatch(ctx, []ExactReadRequest{{Scope: scope, Ref: logical}}); !errors.Is(err, ErrExactRevisionRequired) {
				t.Fatalf("latest alias: %v", err)
			}
			if _, err := repository.ReadExactBatch(ctx, make([]ExactReadRequest, MaxExactReadBatchSize+1)); !errors.Is(err, ErrPayloadTooLarge) {
				t.Fatalf("oversized batch: %v", err)
			}
			if backend == BlobFilesystem {
				var key string
				if err := pool.QueryRow(ctx, `SELECT object_key FROM artifact_blobs WHERE size_bytes = $1`, len("foreign")).Scan(&key); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(blobPath, key), []byte("corrupt"), 0600); err != nil {
					t.Fatal(err)
				}
				if result, err := repository.ReadExactBatch(ctx, []ExactReadRequest{{Scope: foreignScope, Ref: foreign.Ref}}); result != nil || !errors.Is(err, ErrArtifactIntegrity) {
					t.Fatalf("corrupt blob: %v, %v", result, err)
				}
			}
		})
	}
}
