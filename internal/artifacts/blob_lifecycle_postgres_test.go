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

// The registry commit remains successful when unlink fails. Offline cleanup can
// later converge without a cleanup queue or a second logical deletion.
func TestFilesystemFailedUnlinkLeavesCleanableOrphan(t *testing.T) {
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
	ctx = WithBlobRuntime(ctx, NewBlobRuntime(failedUnlinkStore{BlobStore: files}, nil))
	createArtifactRun(t, ctx, pool, "unlink-run", true)
	run, _ := NewService(NewPostgresRepository(pool)).Run("unlink-run")
	if _, err := run.Write(ctx, ArtifactRef{Namespace: "scratch", Name: "orphan"}, Payload{MediaType: "text/plain", Data: []byte("unlink failure")}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := runstore.NewPostgresStore(pool).TransitionRun(ctx, "unlink-run", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
		t.Fatal(err)
	}
	if err := runstore.NewPostgresStore(pool).DeleteReleasedTerminalRun(ctx, "user-1", "unlink-run"); err != nil {
		t.Fatalf("unlink failed logical deletion: %v", err)
	}
	report, err := CleanupFilesystemBlobs(ctx, pool, path, true)
	if err != nil || report.Referenced != 0 || report.Orphans != 1 || report.Removed != 1 {
		t.Fatalf("orphan cleanup: %+v %v", report, err)
	}
}

type failedUnlinkStore struct{ BlobStore }

func (failedUnlinkStore) Delete(context.Context, BlobObject) error { return os.ErrPermission }

// Inject response loss after PostgreSQL actually completed the write. The
// caller sees an ambiguous error; eagerly unlinking its candidate would lose
// referenced content. A CAS retry remains a conflict with one durable revision.
func TestFilesystemLostWriteAcknowledgementKeepsCommittedBytes(t *testing.T) {
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
	broken, _ := NewService(NewPostgresRepository(lostAcknowledgementDB{DBTX: pool})).User("user-1")
	target := ArtifactRef{Namespace: "docs", Name: "committed"}
	payload := Payload{MediaType: "text/plain", Data: []byte("committed before connection loss")}
	if _, err := broken.Write(ctx, target, payload, nil); !errors.Is(err, errLostAcknowledgement) {
		t.Fatalf("write failure: %v", err)
	}
	normal, _ := NewService(NewPostgresRepository(pool)).User("user-1")
	read, err := normal.Read(ctx, target)
	if err != nil || string(read.Payload.Data) != string(payload.Data) {
		t.Fatalf("committed bytes lost: %v", err)
	}
	if _, err := normal.Write(ctx, target, payload, nil); err == nil {
		t.Fatal("CAS retry created another revision")
	}
	var revisions int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_binding_revisions`).Scan(&revisions); err != nil || revisions != 1 {
		t.Fatalf("revisions=%d err=%v", revisions, err)
	}
	report, err := CleanupFilesystemBlobs(ctx, pool, path, true)
	if err != nil || report.Referenced != 1 || report.Missing != 0 || report.Orphans != 1 {
		t.Fatalf("ambiguous write cleanup: %+v %v", report, err)
	}
	if _, err := normal.Read(ctx, read.Ref); err != nil {
		t.Fatal(err)
	}
}

var errLostAcknowledgement = errors.New("injected lost write acknowledgement")

type lostAcknowledgementDB struct{ postgres.DBTX }

func (db lostAcknowledgementDB) QueryRow(ctx context.Context, sql string, args ...any) pgx.Row {
	return lostAcknowledgementRow{Row: db.DBTX.QueryRow(ctx, sql, args...)}
}

type lostAcknowledgementRow struct{ pgx.Row }

func (row lostAcknowledgementRow) Scan(dest ...any) error {
	if err := row.Row.Scan(dest...); err != nil {
		return err
	}
	return errLostAcknowledgement
}
