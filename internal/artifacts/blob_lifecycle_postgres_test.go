package artifacts_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

func TestFilesystemRegistryAndCommitCleanup(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
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
	createArtifactRun(t, ctx, pool, "files-run", true)
	run, _ := NewService(NewPostgresRepository(pool)).Run("files-run")
	written, err := run.Write(ctx, ArtifactRef{Namespace: "scratch", Name: "content"}, Payload{MediaType: "text/plain", Data: []byte("original")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var key string
	var inline bool
	if err := pool.QueryRow(ctx, `SELECT object_key,payload IS NOT NULL FROM artifact_blobs`).Scan(&key, &inline); err != nil {
		t.Fatal(err)
	}
	if inline {
		t.Fatal("filesystem payload duplicated in PostgreSQL")
	}
	if _, err := runstore.NewPostgresStore(pool).TransitionRun(ctx, "files-run", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
		t.Fatal(err)
	}
	abort := errors.New("abort")
	err = postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := runstore.NewPostgresStore(tx).DeleteReleasedTerminalRun(ctx, "user-1", "files-run"); err != nil {
			return err
		}
		return abort
	})
	if !errors.Is(err, abort) {
		t.Fatal(err)
	}
	if _, err := run.Read(ctx, written.Ref); err != nil {
		t.Fatalf("rollback lost bytes: %v", err)
	}
	if err := runstore.NewPostgresStore(pool).DeleteReleasedTerminalRun(ctx, "user-1", "files-run"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(path, key)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("committed unlink: %v", err)
	}
}

func TestFilesystemMissingContentKeepsMetadata(t *testing.T) {
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
	written, err := store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "lost"}, Payload{MediaType: "text/plain", Data: []byte("payload")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var key string
	if err := pool.QueryRow(ctx, `SELECT object_key FROM artifact_blobs`).Scan(&key); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(path, key)); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Read(ctx, written.Ref); !errors.Is(err, ErrBlobMissing) {
		t.Fatalf("lost content: %v", err)
	}
	if _, err := store.Metadata(ctx, written.Ref); err != nil {
		t.Fatalf("lost metadata: %v", err)
	}
}
