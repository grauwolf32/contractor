package postgres_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

func TestFilesystemRegistryAndCommitCleanup(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	path := t.TempDir()
	files, err := filesystem.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	artifactRuntime = blob.NewRuntime(files, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	createArtifactRun(t, ctx, pool, "files-run", true)
	run, _ := NewService(artifactStorage.Repository(pool)).Run("files-run")
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
	if _, err := runstore.NewPostgresStore(pool, artifactStorage).TransitionRun(ctx, "files-run", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
		t.Fatal(err)
	}
	abort := errors.New("abort")
	err = postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := rundeletion.New(tx, artifactStorage).DeleteReleasedTerminalRun(ctx, "user-1", "files-run"); err != nil {
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
	if err := rundeletion.New(pool, artifactStorage).DeleteReleasedTerminalRun(ctx, "user-1", "files-run"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(path, key)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("committed unlink: %v", err)
	}
}

// Fresh operation contexts retain the process backend for all payload paths.
func TestFilesystemBlobUsesInjectedRuntimeWithFreshContext(t *testing.T) {
	ctx := t.Context()
	pool := isolatedArtifactPool(t, ctx)
	path := t.TempDir()
	files, err := filesystem.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	storage := artifactpgtest.NewConfigured(blob.NewRuntime(files, nil), artifactpostgres.DefaultHistoryPolicy())
	createArtifactRun(t, ctx, pool, "fresh-context-run", true)
	run, _ := NewService(storage.Repository(pool)).Run("fresh-context-run")
	fresh := context.Background()
	written, err := run.Write(fresh, ArtifactRef{Namespace: "scratch", Name: "content"}, Payload{MediaType: "text/plain", Data: []byte("filesystem bytes")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	read, err := run.Read(context.Background(), written.Ref)
	if err != nil || string(read.Payload.Data) != "filesystem bytes" {
		t.Fatalf("fresh read: %q %v", read.Payload.Data, err)
	}
	scope, _ := RunScope("fresh-context-run")
	batch, err := storage.Repository(pool).ReadExactBatch(context.Background(), []artifactpostgres.ExactReadRequest{{Scope: scope, Ref: written.Ref}})
	if err != nil || len(batch) != 1 || string(batch[0].Payload.Data) != "filesystem bytes" {
		t.Fatalf("fresh batch: %v", err)
	}
	prepared, err := storage.PreparePayload(context.Background(), pool, Payload{MediaType: "text/plain", Data: []byte("prepared bytes")})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := run.Write(context.Background(), ArtifactRef{Namespace: "scratch", Name: "prepared"}, prepared, nil); err != nil {
		t.Fatal(err)
	}
	keys := blobObjectKeys(t, ctx, pool, `SELECT object_key FROM artifact_blobs WHERE object_key IS NOT NULL`)
	if len(keys) != 2 {
		t.Fatalf("filesystem keys: %v", keys)
	}
	if _, err := runstore.NewPostgresStore(pool, storage).TransitionRun(ctx, "fresh-context-run", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
		t.Fatal(err)
	}
	if err := rundeletion.New(pool, storage).DeleteReleasedTerminalRun(context.Background(), "user-1", "fresh-context-run"); err != nil {
		t.Fatal(err)
	}
	for _, key := range keys {
		if _, err := os.Stat(filepath.Join(path, key)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("committed unlink: %v", err)
		}
	}
}

func TestFilesystemMissingContentKeepsMetadata(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	path := t.TempDir()
	files, err := filesystem.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	artifactRuntime = blob.NewRuntime(files, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	store, _ := NewService(artifactStorage.Repository(pool)).User("user-1")
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
	if _, err := store.Read(ctx, written.Ref); !errors.Is(err, blob.ErrMissing) {
		t.Fatalf("lost content: %v", err)
	}
	if _, err := store.Metadata(ctx, written.Ref); err != nil {
		t.Fatalf("lost metadata: %v", err)
	}
}

// The registry commit remains successful when unlink fails. Offline cleanup can
// later converge without a cleanup queue or a second logical deletion.
func TestFilesystemFailedUnlinkLeavesCleanableOrphan(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	path := t.TempDir()
	files, err := filesystem.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	artifactRuntime = blob.NewRuntime(failedUnlinkStore{Storage: files}, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	createArtifactRun(t, ctx, pool, "unlink-run", true)
	run, _ := NewService(artifactStorage.Repository(pool)).Run("unlink-run")
	if _, err := run.Write(ctx, ArtifactRef{Namespace: "scratch", Name: "orphan"}, Payload{MediaType: "text/plain", Data: []byte("unlink failure")}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := runstore.NewPostgresStore(pool, artifactStorage).TransitionRun(ctx, "unlink-run", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
		t.Fatal(err)
	}
	if err := rundeletion.New(pool, artifactStorage).DeleteReleasedTerminalRun(ctx, "user-1", "unlink-run"); err != nil {
		t.Fatalf("unlink failed logical deletion: %v", err)
	}
	report, err := artifactpostgres.CleanupFilesystemBlobs(ctx, pool, path, true)
	if err != nil || report.Referenced != 0 || report.Orphans != 1 || report.Removed != 1 {
		t.Fatalf("orphan cleanup: %+v %v", report, err)
	}
}

type failedUnlinkStore struct{ blob.Storage }

func (failedUnlinkStore) Delete(context.Context, blob.Object) error { return os.ErrPermission }

func TestFilesystemCreateOnlyConflictDiscardsCandidate(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	path := t.TempDir()
	files, err := filesystem.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	artifactRuntime = blob.NewRuntime(files, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	store, _ := NewService(artifactStorage.Repository(pool)).User("user-1")
	target := ArtifactRef{Namespace: "docs", Name: "immutable"}
	first := Payload{MediaType: "text/plain", Data: []byte("first bytes")}
	if _, err := store.Write(ctx, target, first, nil); err != nil {
		t.Fatal(err)
	}
	for attempt := 0; attempt < 3; attempt++ {
		if _, err := store.Write(ctx, target, Payload{MediaType: "text/plain", Data: []byte("other bytes")}, nil); !errors.Is(err, ErrArtifactConflict) {
			t.Fatalf("create-only conflict %d = %v", attempt, err)
		}
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	txStore, _ := NewService(artifactStorage.Repository(tx)).User("user-1")
	if _, err := txStore.Write(ctx, target, first, nil); !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("transactional create-only conflict = %v", err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	prepared, err := artifactStorage.PreparePayload(ctx, pool, Payload{MediaType: "text/plain", Data: []byte("shared prepared bytes")})
	if err != nil {
		t.Fatal(err)
	}
	preparedTarget := ArtifactRef{Namespace: "docs", Name: "prepared"}
	if _, err := store.Write(ctx, preparedTarget, prepared, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Write(ctx, preparedTarget, prepared, nil); !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("preprepared create-only conflict = %v", err)
	}
	report, err := artifactpostgres.CleanupFilesystemBlobs(ctx, pool, path, false)
	if err != nil || report.Referenced != 2 || report.Orphans != 0 || report.Missing != 0 {
		t.Fatalf("create-only conflict blob cleanup = (%+v, %v)", report, err)
	}
	for _, check := range []struct {
		ref  ArtifactRef
		want string
	}{{target, "first bytes"}, {preparedTarget, "shared prepared bytes"}} {
		read, err := store.Read(ctx, check.ref)
		if err != nil || string(read.Payload.Data) != check.want {
			t.Fatalf("retained conflict winner = (%+v, %v)", read, err)
		}
	}
}

// Inject response loss after PostgreSQL actually completed the write. The
// caller sees an ambiguous error; eagerly unlinking its candidate would lose
// referenced content. A CAS retry remains a conflict with one durable revision
// and discards only its own unpublished candidate.
func TestFilesystemLostWriteAcknowledgementKeepsCommittedBytes(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	path := t.TempDir()
	files, err := filesystem.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	artifactRuntime = blob.NewRuntime(files, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	broken, _ := NewService(artifactStorage.Repository(lostAcknowledgementDB{DBTX: pool})).User("user-1")
	target := ArtifactRef{Namespace: "docs", Name: "committed"}
	payload := Payload{MediaType: "text/plain", Data: []byte("committed before connection loss")}
	if _, err := broken.Write(ctx, target, payload, nil); !errors.Is(err, errLostAcknowledgement) {
		t.Fatalf("write failure: %v", err)
	}
	normal, _ := NewService(artifactStorage.Repository(pool)).User("user-1")
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
	report, err := artifactpostgres.CleanupFilesystemBlobs(ctx, pool, path, true)
	if err != nil || report.Referenced != 1 || report.Missing != 0 || report.Orphans != 0 {
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
