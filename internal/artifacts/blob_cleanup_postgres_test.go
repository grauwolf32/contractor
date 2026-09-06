package artifacts_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
)

func TestFilesystemOfflineCleanupPreservesReferences(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	if err := ClaimBlobBackend(ctx, pool, BlobFilesystem); err != nil {
		t.Fatal(err)
	}
	path := t.TempDir()
	files, err := OpenFilesystemBlobStore(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	ctx = WithBlobRuntime(ctx, NewBlobRuntime(files, nil))
	store, _ := NewService(NewPostgresRepository(pool)).User("user-1")
	keep, err := store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "keep"}, Payload{MediaType: "text/plain", Data: []byte("keep")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	lost, err := store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "lost"}, Payload{MediaType: "text/plain", Data: []byte("lost")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var lostKey string
	if err := pool.QueryRow(ctx, `SELECT object_key FROM artifact_blobs WHERE sha256=sha256('lost'::bytea)`).Scan(&lostKey); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(path, lostKey)); err != nil {
		t.Fatal(err)
	}
	orphan, err := files.Store(ctx, []byte("orphan"))
	if err != nil {
		t.Fatal(err)
	}
	staging := filepath.Join(path, ".staging-00000000000000000000000000000000")
	if err := os.WriteFile(staging, []byte("partial"), 0600); err != nil {
		t.Fatal(err)
	}
	dry, err := CleanupFilesystemBlobs(ctx, pool, path, false)
	if err != nil {
		t.Fatal(err)
	}
	if dry.Applied || dry.Referenced != 2 || dry.Missing != 1 || dry.Orphans != 1 || dry.Staging != 1 || dry.Removed != 0 {
		t.Fatalf("dry report: %+v", dry)
	}
	if _, err := os.Stat(filepath.Join(path, orphan.Key)); err != nil {
		t.Fatal("dry-run mutated orphan")
	}
	applied, err := CleanupFilesystemBlobs(ctx, pool, path, true)
	if err != nil {
		t.Fatal(err)
	}
	if applied.Removed != 2 || applied.Missing != 1 {
		t.Fatalf("apply report: %+v", applied)
	}
	if _, err := store.Read(ctx, keep.Ref); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Metadata(ctx, lost.Ref); err != nil {
		t.Fatal("cleanup deleted missing metadata")
	}
	again, err := CleanupFilesystemBlobs(ctx, pool, path, true)
	if err != nil || again.Removed != 0 {
		t.Fatalf("repeat: %+v %v", again, err)
	}
	outside := filepath.Join(t.TempDir(), "outside")
	if err := os.WriteFile(outside, []byte("protected"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, staging); err != nil {
		t.Fatal(err)
	}
	if _, err := CleanupFilesystemBlobs(ctx, pool, path, true); err == nil {
		t.Fatal("symlink accepted")
	}
	data, err := os.ReadFile(outside)
	if err != nil || string(data) != "protected" {
		t.Fatal("cleanup escaped root")
	}
}
