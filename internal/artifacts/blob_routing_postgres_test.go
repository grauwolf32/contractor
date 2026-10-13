package artifacts_test

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

	. "github.com/grauwolf32/contractor/internal/artifacts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func startupBlobContext(t *testing.T, ctx context.Context, mode BlobBackend, path string, threshold int64) (context.Context, BlobBackend) {
	t.Helper()
	backend, store, err := OpenBlobStore(ctx, mode, path, threshold)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.(io.Closer).Close() })
	return WithBlobRuntime(ctx, NewBlobRuntime(store, nil)), backend
}

func storedBlobLocation(t *testing.T, ctx context.Context, pool *pgxpool.Pool, data []byte) (BlobBackend, *string) {
	t.Helper()
	digest := sha256.Sum256(data)
	var backend BlobBackend
	var key *string
	var inline bool
	if err := pool.QueryRow(ctx, "SELECT backend, object_key, payload IS NOT NULL FROM artifact_blobs WHERE sha256=$1", digest[:]).Scan(&backend, &key, &inline); err != nil {
		t.Fatal(err)
	}
	if inline != (backend == BlobPostgres) || (key != nil) != (backend == BlobFilesystem) {
		t.Fatalf("invalid stored blob shape: %s, key=%v, inline=%v", backend, key, inline)
	}
	return backend, key
}

func TestAutoBlobStorageThresholdAndRestart(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	root := filepath.Join(t.TempDir(), "blobs")
	current, backend := startupBlobContext(t, ctx, BlobAuto, root, DefaultBlobInlineMaxBytes)
	if backend != BlobFilesystem {
		t.Fatalf("writable root selected %s", backend)
	}
	user, _ := NewService(NewPostgresRepository(pool)).User("routing-user")
	for _, tc := range []struct {
		name string
		size int
		want BlobBackend
	}{
		{"empty", 0, BlobPostgres},
		{"threshold", 1048576, BlobPostgres},
		{"above", 1048577, BlobFilesystem},
	} {
		data := bytes.Repeat([]byte{0x5a}, tc.size)
		payload, err := PreparePayload(current, Payload{MediaType: "application/octet-stream", Data: data})
		if err != nil {
			t.Fatal(err)
		}
		written, err := user.Write(current, ArtifactRef{Namespace: "docs", Name: tc.name}, payload, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, _ := storedBlobLocation(t, ctx, pool, data)
		if got != tc.want {
			t.Fatalf("%s stored in %s, want %s", tc.name, got, tc.want)
		}
		restarted, selected := startupBlobContext(t, ctx, BlobPostgres, root, DefaultBlobInlineMaxBytes)
		if selected != BlobPostgres {
			t.Fatal("explicit PostgreSQL writer was ignored")
		}
		read, err := user.Read(restarted, written.Ref)
		if err != nil || !bytes.Equal(read.Payload.Data, data) {
			t.Fatalf("%s after restart: %v", tc.name, err)
		}
	}
	pgContext, _ := startupBlobContext(t, ctx, BlobPostgres, root, DefaultBlobInlineMaxBytes)
	data := bytes.Repeat([]byte{0x31}, 1048577)
	written, err := user.Write(pgContext, ArtifactRef{Namespace: "docs", Name: "large-inline"}, Payload{MediaType: "application/octet-stream", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	read, err := user.Read(current, written.Ref)
	if err != nil || !bytes.Equal(read.Payload.Data, data) {
		t.Fatalf("retained inline bytes: %v", err)
	}
}

func TestMixedBlobDedupKeepsExistingLocation(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	root := t.TempDir()
	pgContext, _ := startupBlobContext(t, ctx, BlobPostgres, root, DefaultBlobInlineMaxBytes)
	fileContext, _ := startupBlobContext(t, ctx, BlobFilesystem, root, DefaultBlobInlineMaxBytes)
	user, _ := NewService(NewPostgresRepository(pool)).User("dedup-routing-user")
	for _, tc := range []struct {
		name        string
		first, next context.Context
		want        BlobBackend
		value       byte
	}{
		{"inline-first", pgContext, fileContext, BlobPostgres, 0x11},
		{"file-first", fileContext, pgContext, BlobFilesystem, 0x22},
	} {
		data := bytes.Repeat([]byte{tc.value}, 1048577)
		payload := Payload{MediaType: "application/octet-stream", Data: data}
		first, err := user.Write(tc.first, ArtifactRef{Namespace: "docs", Name: tc.name}, payload, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, originalKey := storedBlobLocation(t, ctx, pool, data)
		next, err := user.Write(tc.next, ArtifactRef{Namespace: "docs", Name: tc.name + "-duplicate"}, payload, nil)
		if err != nil {
			t.Fatalf("%s across write policy: %v", tc.name, err)
		}
		got, key := storedBlobLocation(t, ctx, pool, data)
		if got != tc.want || (originalKey != nil && (key == nil || *key != *originalKey)) {
			t.Fatalf("%s moved its existing location", tc.name)
		}
		for _, ref := range []ArtifactRef{first.Ref, next.Ref} {
			read, err := user.Read(tc.next, ref)
			if err != nil || !bytes.Equal(read.Payload.Data, data) {
				t.Fatalf("%s duplicate is unreadable: %v", tc.name, err)
			}
		}
	}
	report, err := CleanupFilesystemBlobs(ctx, pool, root, false)
	if err != nil || report.Referenced != 1 || report.Orphans != 0 || report.Staging != 0 {
		t.Fatalf("unused cross-backend candidate leaked: %+v, %v", report, err)
	}
}

func TestAutoBlobUnavailableFilesystemPreservesReadErrors(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	root := filepath.Join(t.TempDir(), "blobs")
	initial, _ := startupBlobContext(t, ctx, BlobAuto, root, DefaultBlobInlineMaxBytes)
	user, _ := NewService(NewPostgresRepository(pool)).User("unavailable-routing-user")
	fileData := bytes.Repeat([]byte{0x33}, 1048577)
	file, err := user.Write(initial, ArtifactRef{Namespace: "docs", Name: "file"}, Payload{MediaType: "application/octet-stream", Data: fileData}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(root, root+"-retained"); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(root, []byte("unavailable directory"), 0600); err != nil {
		t.Fatal(err)
	}
	restarted, selected := startupBlobContext(t, ctx, BlobAuto, root, DefaultBlobInlineMaxBytes)
	if selected != BlobPostgres {
		t.Fatalf("unavailable root selected %s", selected)
	}
	data := bytes.Repeat([]byte{0x44}, 1048577)
	written, err := user.Write(restarted, ArtifactRef{Namespace: "docs", Name: "fallback"}, Payload{MediaType: "application/octet-stream", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := storedBlobLocation(t, ctx, pool, data); got != BlobPostgres {
		t.Fatal("large fallback payload was not inline")
	}
	read, err := user.Read(restarted, written.Ref)
	if err != nil || !bytes.Equal(read.Payload.Data, data) {
		t.Fatalf("fallback bytes: %v", err)
	}
	if _, err := user.Read(restarted, file.Ref); !errors.Is(err, ErrBlobIO) {
		t.Fatalf("unavailable file became successful content: %v", err)
	}
	_, err = user.Write(restarted, ArtifactRef{Namespace: "docs", Name: "missing-duplicate"}, Payload{MediaType: "application/octet-stream", Data: fileData}, nil)
	if !errors.Is(err, ErrBlobIO) {
		t.Fatalf("unavailable deduplication published success: %v", err)
	}
	if _, err := user.Metadata(ctx, ArtifactRef{Namespace: "docs", Name: "missing-duplicate"}); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("failed duplicate left metadata: %v", err)
	}
	if _, _, err := OpenBlobStore(ctx, BlobFilesystem, root, DefaultBlobInlineMaxBytes); err == nil {
		t.Fatal("explicit filesystem mode fell back")
	}
}

func TestMixedBlobPurgeRoutesFilesAfterCommit(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	root := t.TempDir()
	current, _ := startupBlobContext(t, ctx, BlobAuto, root, DefaultBlobInlineMaxBytes)
	createArtifactRun(t, current, pool, "mixed-purge-run", true)
	run, _ := NewService(NewPostgresRepository(pool)).Run("mixed-purge-run")
	for _, item := range []struct {
		name string
		data []byte
	}{
		{"inline", []byte("retained inline")},
		{"file", bytes.Repeat([]byte{0x66}, 1048577)},
	} {
		if _, err := run.Write(current, ArtifactRef{Namespace: "scratch", Name: item.name}, Payload{MediaType: "application/octet-stream", Data: item.data}, nil); err != nil {
			t.Fatal(err)
		}
	}
	var key string
	if err := pool.QueryRow(ctx, "SELECT object_key FROM artifact_blobs WHERE backend='filesystem'").Scan(&key); err != nil {
		t.Fatal(err)
	}
	if _, err := runstore.NewPostgresStore(pool).TransitionRun(ctx, "mixed-purge-run", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
		t.Fatal(err)
	}
	deleting, _ := startupBlobContext(t, ctx, BlobPostgres, root, DefaultBlobInlineMaxBytes)
	abort := errors.New("rollback")
	err := postgres.InTx(deleting, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := rundeletion.New(tx).DeleteReleasedTerminalRun(deleting, "user-1", "mixed-purge-run"); err != nil {
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
	if err := rundeletion.New(pool).DeleteReleasedTerminalRun(deleting, "user-1", "mixed-purge-run"); err != nil {
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
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	root := t.TempDir()
	fileContext, _ := startupBlobContext(t, ctx, BlobFilesystem, root, DefaultBlobInlineMaxBytes)
	user, _ := NewService(NewPostgresRepository(pool)).User("mixed-race-user")
	payload := Payload{MediaType: "text/plain", Data: []byte("same digest during a backend change")}
	if _, err := user.Write(fileContext, ArtifactRef{Namespace: "docs", Name: "original"}, payload, nil); err != nil {
		t.Fatal(err)
	}
	_, key := storedBlobLocation(t, ctx, pool, payload.Data)
	_, store, err := OpenBlobStore(ctx, BlobPostgres, root, DefaultBlobInlineMaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer store.(io.Closer).Close()
	paused := pausedBlobRead{BlobStore: store, key: *key, entered: make(chan struct{}, 1), resume: make(chan struct{})}
	delayed := WithBlobRuntime(ctx, NewBlobRuntime(paused, nil))
	target := ArtifactRef{Namespace: "docs", Name: "delayed-inline"}
	finished := make(chan error, 1)
	var release sync.Once
	var writer sync.WaitGroup
	writer.Add(1)
	go func() {
		defer writer.Done()
		_, err := user.Write(delayed, target, payload, nil)
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
	if _, err := user.Write(fileContext, ArtifactRef{Namespace: "docs", Name: "repair"}, payload, nil); err != nil {
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
	if _, err := user.Metadata(ctx, target); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("rejected mixed-backend race left partial metadata: %v", err)
	}
}
