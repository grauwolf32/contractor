package postgres_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
)

func TestReadExactBatchPreservesScopeRevisionAndBlobIntegrity(t *testing.T) {
	for _, backend := range []blob.Backend{blob.Postgres, blob.Filesystem} {
		t.Run(string(backend), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			pool := isolatedArtifactPool(t, ctx)
			var blobPath string
			if backend == blob.Filesystem {
				blobPath = t.TempDir()
				files, err := filesystem.Open(ctx, blobPath)
				if err != nil {
					t.Fatal(err)
				}
				defer files.Close()
				ctx = blob.WithRuntime(ctx, blob.NewRuntime(files, nil))
			}
			repository := artifactpostgres.NewRepository(pool)
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
			requests := []artifactpostgres.ExactReadRequest{{Scope: scope, Ref: latest.Ref}, {Scope: scope, Ref: first.Ref}, {Scope: foreignScope, Ref: foreign.Ref}, {Scope: scope, Ref: first.Ref}}
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
			if _, err := repository.ReadExactBatch(ctx, []artifactpostgres.ExactReadRequest{{Scope: scope, Ref: logical}}); !errors.Is(err, ErrExactRevisionRequired) {
				t.Fatalf("latest alias: %v", err)
			}
			if _, err := repository.ReadExactBatch(ctx, make([]artifactpostgres.ExactReadRequest, MaxExactReadBatchSize+1)); !errors.Is(err, blob.ErrTooLarge) {
				t.Fatalf("oversized batch: %v", err)
			}
			if backend == blob.Filesystem {
				var key string
				if err := pool.QueryRow(ctx, `SELECT object_key FROM artifact_blobs WHERE size_bytes = $1`, len("foreign")).Scan(&key); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(blobPath, key), []byte("corrupt"), 0600); err != nil {
					t.Fatal(err)
				}
				if result, err := repository.ReadExactBatch(ctx, []artifactpostgres.ExactReadRequest{{Scope: foreignScope, Ref: foreign.Ref}}); result != nil || !errors.Is(err, blob.ErrIntegrity) {
					t.Fatalf("corrupt blob: %v, %v", result, err)
				}
			}
		})
	}
}
