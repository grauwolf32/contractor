package artifacts_test

import (
	"bytes"
	"context"
	"errors"
	"log"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

func TestFilesystemRegistryAndCommitCleanup(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
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
		if err := rundeletion.New(tx).DeleteReleasedTerminalRun(ctx, "user-1", "files-run"); err != nil {
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
	if err := rundeletion.New(pool).DeleteReleasedTerminalRun(ctx, "user-1", "files-run"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(path, key)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("committed unlink: %v", err)
	}
}

// A context without the installed blob runtime, such as a fresh
// context.Background(), cannot open or unlink filesystem objects. Reads and
// deletion fail as logged internal errors instead of reporting corrupt content
// or silently orphaning the file, and nothing is deleted until a context with
// the runtime retries.
func TestFilesystemBlobWithoutRuntimeFailsLoudly(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	path := t.TempDir()
	files, err := OpenFilesystemBlobStore(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	installed := WithBlobRuntime(ctx, NewBlobRuntime(files, nil))
	createArtifactRun(t, installed, pool, "runtimeless-run", true)
	run, _ := NewService(NewPostgresRepository(pool)).Run("runtimeless-run")
	written, err := run.Write(installed, ArtifactRef{Namespace: "scratch", Name: "content"}, Payload{MediaType: "text/plain", Data: []byte("filesystem bytes")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var key string
	if err := pool.QueryRow(ctx, `SELECT object_key FROM artifact_blobs`).Scan(&key); err != nil {
		t.Fatal(err)
	}
	logs := captureDefaultLogger(t)
	runtimeless := context.Background()
	loud := func(operation string, err error) {
		t.Helper()
		if !errors.Is(err, ErrBlobRuntimeMissing) || errors.Is(err, ErrArtifactIntegrity) ||
			errors.Is(err, ErrBlobMissing) || errors.Is(err, ErrBlobIO) {
			t.Fatalf("%s without blob runtime = %v, want an internal missing-runtime error", operation, err)
		}
	}
	_, err = run.Read(runtimeless, written.Ref)
	loud("read", err)
	scope, _ := RunScope("runtimeless-run")
	_, err = NewPostgresRepository(pool).ReadExactBatch(runtimeless, []ExactReadRequest{{Scope: scope, Ref: written.Ref}})
	loud("exact batch read", err)
	prepared, err := PreparePayload(installed, Payload{MediaType: "text/plain", Data: []byte("prepared bytes")})
	if err != nil {
		t.Fatal(err)
	}
	_, err = run.Write(runtimeless, ArtifactRef{Namespace: "scratch", Name: "prepared"}, prepared, nil)
	loud("write of a prepared filesystem payload", err)
	if _, err := runstore.NewPostgresStore(pool).TransitionRun(ctx, "runtimeless-run", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
		t.Fatal(err)
	}
	err = rundeletion.New(pool).DeleteReleasedTerminalRun(runtimeless, "user-1", "runtimeless-run")
	loud("Run deletion", err)
	for _, operation := range []string{"operation=read", "operation=check", "operation=delete"} {
		if !strings.Contains(logs.String(), operation) {
			t.Fatalf("missing runtime was not logged for %s: %q", operation, logs.String())
		}
	}
	// The failed deletion rolled back: the binding still reads its bytes.
	read, err := run.Read(installed, written.Ref)
	if err != nil || string(read.Payload.Data) != "filesystem bytes" {
		t.Fatalf("read with runtime after failed deletion = (%q, %v)", read.Payload.Data, err)
	}
	if _, err := os.Stat(filepath.Join(path, key)); err != nil {
		t.Fatalf("failed deletion touched the file: %v", err)
	}
	if err := rundeletion.New(pool).DeleteReleasedTerminalRun(installed, "user-1", "runtimeless-run"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(path, key)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("committed unlink: %v", err)
	}
}

// captureDefaultLogger routes slog.Default to a buffer for one sequential test
// and restores the default logger and the log package output afterwards.
func captureDefaultLogger(t *testing.T) *syncBuffer {
	t.Helper()
	previous, writer, flags := slog.Default(), log.Writer(), log.Flags()
	output := &syncBuffer{}
	slog.SetDefault(slog.New(slog.NewTextHandler(output, nil)))
	t.Cleanup(func() {
		slog.SetDefault(previous)
		log.SetOutput(writer)
		log.SetFlags(flags)
	})
	return output
}

type syncBuffer struct {
	mu     sync.Mutex
	buffer bytes.Buffer
}

func (b *syncBuffer) Write(data []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buffer.Write(data)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buffer.String()
}

func TestFilesystemMissingContentKeepsMetadata(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
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
	if err := rundeletion.New(pool).DeleteReleasedTerminalRun(ctx, "user-1", "unlink-run"); err != nil {
		t.Fatalf("unlink failed logical deletion: %v", err)
	}
	report, err := CleanupFilesystemBlobs(ctx, pool, path, true)
	if err != nil || report.Referenced != 0 || report.Orphans != 1 || report.Removed != 1 {
		t.Fatalf("orphan cleanup: %+v %v", report, err)
	}
}

type failedUnlinkStore struct{ BlobStore }

func (failedUnlinkStore) Delete(context.Context, BlobObject) error { return os.ErrPermission }

func TestFilesystemCreateOnlyConflictDiscardsCandidate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	path := t.TempDir()
	files, err := OpenFilesystemBlobStore(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	ctx = WithBlobRuntime(ctx, NewBlobRuntime(files, nil))
	store, _ := NewService(NewPostgresRepository(pool)).User("user-1")
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
	txStore, _ := NewService(NewPostgresRepository(tx)).User("user-1")
	if _, err := txStore.Write(ctx, target, first, nil); !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("transactional create-only conflict = %v", err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	prepared, err := PreparePayload(ctx, Payload{MediaType: "text/plain", Data: []byte("shared prepared bytes")})
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
	report, err := CleanupFilesystemBlobs(ctx, pool, path, false)
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
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
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
