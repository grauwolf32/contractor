package postgres_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func startupBlobStorage(t *testing.T, ctx context.Context, mode blob.Backend, path string, threshold int64) (*artifactpostgres.Storage, blob.Backend) {
	t.Helper()
	backend, store, err := backends.Open(ctx, mode, path, threshold)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.(io.Closer).Close() })
	return artifactpgtest.NewConfigured(blob.NewRuntime(store, nil), artifactpostgres.DefaultHistoryPolicy()), backend
}

func storedBlobLocation(t *testing.T, ctx context.Context, pool *pgxpool.Pool, data []byte) (blob.Backend, *string) {
	t.Helper()
	digest := sha256.Sum256(data)
	var backend blob.Backend
	var key *string
	var inline bool
	if err := pool.QueryRow(ctx, "SELECT backend, object_key, payload IS NOT NULL FROM artifact_blobs WHERE sha256=$1", digest[:]).Scan(&backend, &key, &inline); err != nil {
		t.Fatal(err)
	}
	if inline != (backend == blob.Postgres) || (key != nil) != (backend == blob.Filesystem) {
		t.Fatalf("invalid stored blob shape: %s, key=%v, inline=%v", backend, key, inline)
	}
	return backend, key
}

func TestAutoBlobStorageThresholdAndRestart(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	root := filepath.Join(t.TempDir(), "blobs")
	current, backend := startupBlobStorage(t, ctx, blob.Auto, root, blob.DefaultInlineMaxBytes)
	if backend != blob.Filesystem {
		t.Fatalf("writable root selected %s", backend)
	}
	for _, tc := range []struct {
		name string
		size int
		want blob.Backend
	}{
		{"empty", 0, blob.Postgres},
		{"threshold", 1048576, blob.Postgres},
		{"above", 1048577, blob.Filesystem},
	} {
		data := bytes.Repeat([]byte{0x5a}, tc.size)
		payload, err := current.PreparePayload(ctx, pool, Payload{MediaType: "application/octet-stream", Data: data})
		if err != nil {
			t.Fatal(err)
		}
		written, err := routingUser(t, current, pool, "routing-user").Write(ctx, ArtifactRef{Namespace: "docs", Name: tc.name}, payload, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, _ := storedBlobLocation(t, ctx, pool, data)
		if got != tc.want {
			t.Fatalf("%s stored in %s, want %s", tc.name, got, tc.want)
		}
		restarted, selected := startupBlobStorage(t, ctx, blob.Postgres, root, blob.DefaultInlineMaxBytes)
		if selected != blob.Postgres {
			t.Fatal("explicit PostgreSQL writer was ignored")
		}
		read, err := routingUser(t, restarted, pool, "routing-user").Read(ctx, written.Ref)
		if err != nil || !bytes.Equal(read.Payload.Data, data) {
			t.Fatalf("%s after restart: %v", tc.name, err)
		}
	}
	pgContext, _ := startupBlobStorage(t, ctx, blob.Postgres, root, blob.DefaultInlineMaxBytes)
	data := bytes.Repeat([]byte{0x31}, 1048577)
	written, err := routingUser(t, pgContext, pool, "routing-user").Write(ctx, ArtifactRef{Namespace: "docs", Name: "large-inline"}, Payload{MediaType: "application/octet-stream", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	read, err := routingUser(t, current, pool, "routing-user").Read(ctx, written.Ref)
	if err != nil || !bytes.Equal(read.Payload.Data, data) {
		t.Fatalf("retained inline bytes: %v", err)
	}
}

func TestMixedBlobDedupKeepsExistingLocation(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	root := t.TempDir()
	pgContext, _ := startupBlobStorage(t, ctx, blob.Postgres, root, blob.DefaultInlineMaxBytes)
	fileContext, _ := startupBlobStorage(t, ctx, blob.Filesystem, root, blob.DefaultInlineMaxBytes)
	for _, tc := range []struct {
		name        string
		first, next *artifactpostgres.Storage
		want        blob.Backend
		value       byte
	}{
		{"inline-first", pgContext, fileContext, blob.Postgres, 0x11},
		{"file-first", fileContext, pgContext, blob.Filesystem, 0x22},
	} {
		data := bytes.Repeat([]byte{tc.value}, 1048577)
		payload := Payload{MediaType: "application/octet-stream", Data: data}
		first, err := routingUser(t, tc.first, pool, "dedup-routing-user").Write(ctx, ArtifactRef{Namespace: "docs", Name: tc.name}, payload, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, originalKey := storedBlobLocation(t, ctx, pool, data)
		next, err := routingUser(t, tc.next, pool, "dedup-routing-user").Write(ctx, ArtifactRef{Namespace: "docs", Name: tc.name + "-duplicate"}, payload, nil)
		if err != nil {
			t.Fatalf("%s across write policy: %v", tc.name, err)
		}
		got, key := storedBlobLocation(t, ctx, pool, data)
		if got != tc.want || (originalKey != nil && (key == nil || *key != *originalKey)) {
			t.Fatalf("%s moved its existing location", tc.name)
		}
		for _, ref := range []ArtifactRef{first.Ref, next.Ref} {
			read, err := routingUser(t, tc.next, pool, "dedup-routing-user").Read(ctx, ref)
			if err != nil || !bytes.Equal(read.Payload.Data, data) {
				t.Fatalf("%s duplicate is unreadable: %v", tc.name, err)
			}
		}
	}
	report, err := artifactpostgres.CleanupFilesystemBlobs(ctx, pool, root, false)
	if err != nil || report.Referenced != 1 || report.Orphans != 0 || report.Staging != 0 {
		t.Fatalf("unused cross-backend candidate leaked: %+v, %v", report, err)
	}
}

func TestAutoBlobUnavailableFilesystemPreservesReadErrors(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	root := filepath.Join(t.TempDir(), "blobs")
	initial, _ := startupBlobStorage(t, ctx, blob.Auto, root, blob.DefaultInlineMaxBytes)
	fileData := bytes.Repeat([]byte{0x33}, 1048577)
	file, err := routingUser(t, initial, pool, "unavailable-routing-user").Write(ctx, ArtifactRef{Namespace: "docs", Name: "file"}, Payload{MediaType: "application/octet-stream", Data: fileData}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(root, root+"-retained"); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(root, []byte("unavailable directory"), 0600); err != nil {
		t.Fatal(err)
	}
	restarted, selected := startupBlobStorage(t, ctx, blob.Auto, root, blob.DefaultInlineMaxBytes)
	if selected != blob.Postgres {
		t.Fatalf("unavailable root selected %s", selected)
	}
	data := bytes.Repeat([]byte{0x44}, 1048577)
	written, err := routingUser(t, restarted, pool, "unavailable-routing-user").Write(ctx, ArtifactRef{Namespace: "docs", Name: "fallback"}, Payload{MediaType: "application/octet-stream", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := storedBlobLocation(t, ctx, pool, data); got != blob.Postgres {
		t.Fatal("large fallback payload was not inline")
	}
	read, err := routingUser(t, restarted, pool, "unavailable-routing-user").Read(ctx, written.Ref)
	if err != nil || !bytes.Equal(read.Payload.Data, data) {
		t.Fatalf("fallback bytes: %v", err)
	}
	if _, err := routingUser(t, restarted, pool, "unavailable-routing-user").Read(ctx, file.Ref); !errors.Is(err, blob.ErrIO) {
		t.Fatalf("unavailable file became successful content: %v", err)
	}
	_, err = routingUser(t, restarted, pool, "unavailable-routing-user").Write(ctx, ArtifactRef{Namespace: "docs", Name: "missing-duplicate"}, Payload{MediaType: "application/octet-stream", Data: fileData}, nil)
	if !errors.Is(err, blob.ErrIO) {
		t.Fatalf("unavailable deduplication published success: %v", err)
	}
	if _, err := routingUser(t, initial, pool, "unavailable-routing-user").Metadata(ctx, ArtifactRef{Namespace: "docs", Name: "missing-duplicate"}); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("failed duplicate left metadata: %v", err)
	}
	if _, _, err := backends.Open(ctx, blob.Filesystem, root, blob.DefaultInlineMaxBytes); err == nil {
		t.Fatal("explicit filesystem mode fell back")
	}
}

func TestMixedBlobPurgeRoutesFilesAfterCommit(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	root := t.TempDir()
	current, _ := startupBlobStorage(t, ctx, blob.Auto, root, blob.DefaultInlineMaxBytes)
	createArtifactRun(t, ctx, pool, "mixed-purge-run", true)
	run, _ := NewService(current.Repository(pool)).Run("mixed-purge-run")
	for _, item := range []struct {
		name string
		data []byte
	}{
		{"inline", []byte("retained inline")},
		{"file", bytes.Repeat([]byte{0x66}, 1048577)},
	} {
		if _, err := run.Write(ctx, ArtifactRef{Namespace: "scratch", Name: item.name}, Payload{MediaType: "application/octet-stream", Data: item.data}, nil); err != nil {
			t.Fatal(err)
		}
	}
	var key string
	if err := pool.QueryRow(ctx, "SELECT object_key FROM artifact_blobs WHERE backend='filesystem'").Scan(&key); err != nil {
		t.Fatal(err)
	}
	if _, err := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()).TransitionRun(ctx, "mixed-purge-run", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
		t.Fatal(err)
	}
	deleting, _ := startupBlobStorage(t, ctx, blob.Postgres, root, blob.DefaultInlineMaxBytes)
	abort := errors.New("rollback")
	err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := rundeletion.New(tx, deleting).DeleteReleasedTerminalRun(ctx, "user-1", "mixed-purge-run"); err != nil {
			return err
		}
		return abort
	})
	if !errors.Is(err, abort) {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, key)); err != nil {
		t.Fatalf("rollback removed file: %v", err)
	}
	if err := rundeletion.New(pool, deleting).DeleteReleasedTerminalRun(ctx, "user-1", "mixed-purge-run"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, key)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("committed purge retained file: %v", err)
	}
	var count int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM artifact_blobs").Scan(&count); err != nil || count != 0 {
		t.Fatalf("mixed purge retained %d blobs: %v", count, err)
	}
}

func TestMixedDedupRejectsChangedUnverifiedFileKey(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	root := t.TempDir()
	fileContext, _ := startupBlobStorage(t, ctx, blob.Filesystem, root, blob.DefaultInlineMaxBytes)
	payload := Payload{MediaType: "text/plain", Data: []byte("same digest during a backend change")}
	if _, err := routingUser(t, fileContext, pool, "mixed-race-user").Write(ctx, ArtifactRef{Namespace: "docs", Name: "original"}, payload, nil); err != nil {
		t.Fatal(err)
	}
	_, key := storedBlobLocation(t, ctx, pool, payload.Data)
	_, store, err := backends.Open(ctx, blob.Postgres, root, blob.DefaultInlineMaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer store.(io.Closer).Close()
	paused := pausedBlobRead{Storage: store, key: *key, entered: make(chan struct{}, 1), resume: make(chan struct{})}
	artifactRuntime = blob.NewRuntime(paused, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	target := ArtifactRef{Namespace: "docs", Name: "delayed-inline"}
	finished := make(chan error, 1)
	var release sync.Once
	var writer sync.WaitGroup
	writer.Add(1)
	go func() {
		defer writer.Done()
		_, err := routingUser(t, artifactStorage, pool, "mixed-race-user").Write(ctx, target, payload, nil)
		finished <- err
	}()
	defer func() { release.Do(func() { close(paused.resume) }); writer.Wait() }()
	select {
	case <-paused.entered:
	case <-ctx.Done():
		t.Fatal("inline writer did not verify the original file")
	}
	if err := os.Remove(filepath.Join(root, *key)); err != nil {
		t.Fatal(err)
	}
	if _, err := routingUser(t, fileContext, pool, "mixed-race-user").Write(ctx, ArtifactRef{Namespace: "docs", Name: "repair"}, payload, nil); err != nil {
		t.Fatal(err)
	}
	_, replacement := storedBlobLocation(t, ctx, pool, payload.Data)
	if *replacement == *key {
		t.Fatal("competing writer did not replace the missing generation")
	}
	if err := os.Remove(filepath.Join(root, *replacement)); err != nil {
		t.Fatal(err)
	}
	release.Do(func() { close(paused.resume) })
	if err := <-finished; err == nil {
		t.Fatal("inline writer acknowledged an unverified replacement key")
	}
	if _, err := routingUser(t, artifactStorage, pool, "mixed-race-user").Metadata(ctx, target); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("rejected mixed-backend race left partial metadata: %v", err)
	}
}

func routingUser(t *testing.T, storage *artifactpostgres.Storage, pool *pgxpool.Pool, owner string) ScopedStore {
	t.Helper()
	user, err := NewService(storage.Repository(pool)).User(owner)
	if err != nil {
		t.Fatal(err)
	}
	return user
}
