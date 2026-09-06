package artifacts_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestFilesystemDedupRepairsMissingContent(t *testing.T) {
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
	store, _ := NewService(NewPostgresRepository(pool)).User("user-1")
	payload := Payload{MediaType: "text/plain", Data: []byte("deduplicated payload")}
	first, err := store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "first"}, payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	var oldKey string
	if err := pool.QueryRow(ctx, `SELECT object_key FROM artifact_blobs`).Scan(&oldKey); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(path, oldKey)); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Read(ctx, first.Ref); !errors.Is(err, ErrBlobMissing) {
		t.Fatalf("expected missing content before repair: %v", err)
	}
	second, err := store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "second"}, payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, ref := range []ArtifactRef{second.Ref, first.Ref} {
		read, err := store.Read(ctx, ref)
		if err != nil || string(read.Payload.Data) != string(payload.Data) {
			t.Fatalf("successful deduplicated write left unreadable revision: %v", err)
		}
	}
	report, err := CleanupFilesystemBlobs(ctx, pool, path, false)
	if err != nil || report.Referenced != 1 || report.Missing != 0 || report.Orphans != 0 {
		t.Fatalf("repair registry/files: %+v %v", report, err)
	}
}

type dedupFixture struct {
	ctx     context.Context
	pool    *pgxpool.Pool
	path    string
	files   *FilesystemBlobStore
	store   ScopedStore
	payload Payload
	first   WriteResult
	key     string
}

func newDedupFixture(t *testing.T) dedupFixture {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	pool := isolatedArtifactPool(t, ctx)
	if err := ClaimBlobBackend(ctx, pool, BlobFilesystem); err != nil {
		t.Fatal(err)
	}
	path := t.TempDir()
	files, err := OpenFilesystemBlobStore(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = files.Close() })
	ctx = WithBlobRuntime(ctx, NewBlobRuntime(files, nil))
	store, _ := NewService(NewPostgresRepository(pool)).User("user-1")
	payload := Payload{MediaType: "text/plain", Data: []byte("deduplicated payload")}
	first, err := store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "first"}, payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	var key string
	if err := pool.QueryRow(ctx, `SELECT object_key FROM artifact_blobs`).Scan(&key); err != nil {
		t.Fatal(err)
	}
	return dedupFixture{ctx, pool, path, files, store, payload, first, key}
}

func TestFilesystemDedupHealthyKeyIsReused(t *testing.T) {
	f := newDedupFixture(t)
	written, err := f.store.Write(f.ctx, ArtifactRef{Namespace: "docs", Name: "second"}, f.payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	var key string
	if err := f.pool.QueryRow(f.ctx, `SELECT object_key FROM artifact_blobs`).Scan(&key); err != nil || key != f.key {
		t.Fatalf("healthy key changed: %v", err)
	}
	if _, err := f.store.Read(f.ctx, written.Ref); err != nil {
		t.Fatal(err)
	}
	report, err := CleanupFilesystemBlobs(f.ctx, f.pool, f.path, false)
	if err != nil || report.Referenced != 1 || report.Orphans != 0 || report.Missing != 0 {
		t.Fatalf("healthy dedup: %+v %v", report, err)
	}
}

func TestFilesystemDedupCASUpdateRepairsBothRevisions(t *testing.T) {
	f := newDedupFixture(t)
	if err := os.Remove(filepath.Join(f.path, f.key)); err != nil {
		t.Fatal(err)
	}
	written, err := f.store.Write(f.ctx, ArtifactRef{Namespace: "docs", Name: "first"}, f.payload, f.first.Ref.Revision)
	if err != nil {
		t.Fatal(err)
	}
	if *written.Ref.Revision == *f.first.Ref.Revision {
		t.Fatal("CAS update reused logical revision")
	}
	for _, ref := range []ArtifactRef{f.first.Ref, written.Ref, {Namespace: "docs", Name: "first"}} {
		read, err := f.store.Read(f.ctx, ref)
		if err != nil || string(read.Payload.Data) != string(f.payload.Data) {
			t.Fatalf("CAS repair: %v", err)
		}
	}
}

func TestFilesystemDedupAuditWriteRepairsMissingContent(t *testing.T) {
	f := newDedupFixture(t)
	if err := os.Remove(filepath.Join(f.path, f.key)); err != nil {
		t.Fatal(err)
	}
	_, _, err := projectstore.NewPostgresStore(f.pool).Create(f.ctx, projectstore.CreateParams{
		ProjectID: "dedup-audit", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Audit", IdempotencyKey: "dedup-audit", RequestDigest: "sha256:" + strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	written, err := NewPostgresRepository(f.pool).WriteAuditArtifact(f.ctx, mustProjectScope(t, "dedup-audit"), ArtifactRef{Namespace: "audit-test", Name: "repaired"}, f.payload)
	if err != nil {
		t.Fatal(err)
	}
	project, _ := NewService(NewPostgresRepository(f.pool)).Project("dedup-audit")
	if read, err := project.Read(f.ctx, written.Ref); err != nil || string(read.Payload.Data) != string(f.payload.Data) {
		t.Fatalf("Audit repair: %v", err)
	}
	if _, err := f.store.Read(f.ctx, f.first.Ref); err != nil {
		t.Fatalf("old revision not repaired: %v", err)
	}
}

func TestFilesystemDedupRepairRollsBackWithCallerTransaction(t *testing.T) {
	f := newDedupFixture(t)
	if err := os.Remove(filepath.Join(f.path, f.key)); err != nil {
		t.Fatal(err)
	}
	rollback := errors.New("rollback repair")
	err := postgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		store, _ := NewService(NewPostgresRepository(tx)).User("user-1")
		written, err := store.Write(f.ctx, ArtifactRef{Namespace: "docs", Name: "rolled-back"}, f.payload, nil)
		if err != nil {
			return err
		}
		if _, err := store.Read(f.ctx, written.Ref); err != nil {
			return err
		}
		return rollback
	})
	if !errors.Is(err, rollback) {
		t.Fatal(err)
	}
	var key string
	var revisions int
	if err := f.pool.QueryRow(f.ctx, `SELECT object_key, (SELECT count(*) FROM artifact_binding_revisions) FROM artifact_blobs`).Scan(&key, &revisions); err != nil || key != f.key || revisions != 1 {
		t.Fatalf("repair escaped rollback: %d %v", revisions, err)
	}
	if _, err := f.store.Read(f.ctx, f.first.Ref); !errors.Is(err, ErrBlobMissing) {
		t.Fatalf("rollback changed old content: %v", err)
	}
}

func TestFilesystemDedupCorruptionDoesNotPublishOrOverwrite(t *testing.T) {
	f := newDedupFixture(t)
	corrupt := []byte(strings.Repeat("x", len(f.payload.Data)))
	if err := os.WriteFile(filepath.Join(f.path, f.key), corrupt, 0600); err != nil {
		t.Fatal(err)
	}
	_, err := f.store.Write(f.ctx, ArtifactRef{Namespace: "docs", Name: "rejected"}, f.payload, nil)
	if !errors.Is(err, ErrArtifactIntegrity) {
		t.Fatalf("corruption hidden: %v", err)
	}
	var revisions int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_binding_revisions`).Scan(&revisions); err != nil || revisions != 1 {
		t.Fatalf("failed verification published: %d %v", revisions, err)
	}
	if data, err := os.ReadFile(filepath.Join(f.path, f.key)); err != nil || string(data) != string(corrupt) {
		t.Fatalf("corruption overwritten: %v", err)
	}
}

func TestFilesystemDedupMissingPreparedCandidateIsNotPublished(t *testing.T) {
	f := newDedupFixture(t)
	payload := Payload{MediaType: "text/plain", Data: []byte("prepared candidate")}
	prepared, err := PreparePayload(f.ctx, payload)
	if err != nil {
		t.Fatal(err)
	}
	// This fixture owns its temporary root. Remove only the unreferenced candidate.
	err = filepath.WalkDir(f.path, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && path != filepath.Join(f.path, f.key) {
			return os.Remove(path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.store.Write(f.ctx, ArtifactRef{Namespace: "docs", Name: "not-published"}, prepared, nil)
	if !errors.Is(err, ErrBlobMissing) {
		t.Fatalf("missing candidate published: %v", err)
	}
	var revisions int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_binding_revisions`).Scan(&revisions); err != nil || revisions != 1 {
		t.Fatalf("unexpected revision: %d %v", revisions, err)
	}
}

type pausedBlobRead struct {
	BlobStore
	key     string
	entered chan struct{}
	resume  chan struct{}
}

func (s pausedBlobRead) Read(ctx context.Context, object BlobObject) ([]byte, error) {
	data, err := s.BlobStore.Read(ctx, object)
	if object.Key == s.key {
		s.entered <- struct{}{}
		select {
		case <-s.resume:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return data, err
}

func TestFilesystemDedupChangedKeyCannotBeReusedWithoutVerification(t *testing.T) {
	f := newDedupFixture(t)
	paused := pausedBlobRead{BlobStore: f.files, key: f.key, entered: make(chan struct{}, 1), resume: make(chan struct{})}
	ctx := WithBlobRuntime(f.ctx, NewBlobRuntime(paused, nil))
	finished := make(chan error, 1)
	go func() {
		_, err := f.store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "delayed"}, f.payload, nil)
		finished <- err
	}()
	defer func() {
		close(paused.resume)
		if err := <-finished; err != nil {
			t.Error(err)
		}
	}()
	select {
	case <-paused.entered:
	case <-f.ctx.Done():
		t.Fatal("writer did not reach physical verification")
	}
	if err := os.Remove(filepath.Join(f.path, f.key)); err != nil {
		t.Fatal(err)
	}
	if _, err := f.store.Write(f.ctx, ArtifactRef{Namespace: "docs", Name: "concurrent"}, f.payload, nil); err != nil {
		t.Fatal(err)
	}
	var replacement string
	if err := f.pool.QueryRow(f.ctx, `SELECT object_key FROM artifact_blobs`).Scan(&replacement); err != nil {
		t.Fatal(err)
	}
	if replacement == f.key {
		t.Fatal("missing old key was not replaced")
	}
	if err := os.Remove(filepath.Join(f.path, replacement)); err != nil {
		t.Fatal(err)
	}
	// The delayed writer checked the old key. It must not reuse the different,
	// now-missing key installed by the competing publication.
	t.Cleanup(func() {
		for _, name := range []string{"first", "concurrent", "delayed"} {
			read, err := f.store.Read(f.ctx, ArtifactRef{Namespace: "docs", Name: name})
			if err != nil || string(read.Payload.Data) != string(f.payload.Data) {
				t.Errorf("concurrent dedup left %s unreadable: %v", name, err)
			}
		}
	})
}

func TestFilesystemDedupConcurrentMissingContentRepairsRemainReadable(t *testing.T) {
	f := newDedupFixture(t)
	if err := os.Remove(filepath.Join(f.path, f.key)); err != nil {
		t.Fatal(err)
	}
	paused := pausedBlobRead{BlobStore: f.files, key: f.key, entered: make(chan struct{}, 2), resume: make(chan struct{})}
	ctx := WithBlobRuntime(f.ctx, NewBlobRuntime(paused, nil))
	var release sync.Once
	var writers sync.WaitGroup
	finished := make(chan error, 2)
	defer func() { release.Do(func() { close(paused.resume) }); writers.Wait() }()
	for _, name := range []string{"second", "third"} {
		writers.Add(1)
		go func() {
			defer writers.Done()
			_, err := f.store.Write(ctx, ArtifactRef{Namespace: "docs", Name: name}, f.payload, nil)
			finished <- err
		}()
	}
	for range 2 {
		select {
		case <-paused.entered:
		case <-f.ctx.Done():
			t.Fatal("writers did not both verify missing content")
		}
	}
	release.Do(func() { close(paused.resume) })
	for range 2 {
		if err := <-finished; err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"first", "second", "third"} {
		read, err := f.store.Read(f.ctx, ArtifactRef{Namespace: "docs", Name: name})
		if err != nil || string(read.Payload.Data) != string(f.payload.Data) {
			t.Fatalf("concurrent repair %s: %v", name, err)
		}
	}
	report, err := CleanupFilesystemBlobs(f.ctx, f.pool, f.path, false)
	if err != nil || report.Referenced != 1 || report.Missing != 0 || report.Orphans != 1 {
		t.Fatalf("concurrent candidate retention: %+v %v", report, err)
	}
}

type lostRepairAcknowledgementDB struct{ postgres.DBTX }

func (db lostRepairAcknowledgementDB) QueryRow(ctx context.Context, sql string, args ...any) pgx.Row {
	row := db.DBTX.QueryRow(ctx, sql, args...)
	if strings.Contains(sql, "WITH scope_ready") {
		return lostAcknowledgementRow{Row: row}
	}
	return row
}

func TestFilesystemDedupLostRepairAcknowledgementKeepsCommittedCandidate(t *testing.T) {
	f := newDedupFixture(t)
	if err := os.Remove(filepath.Join(f.path, f.key)); err != nil {
		t.Fatal(err)
	}
	store, _ := NewService(NewPostgresRepository(lostRepairAcknowledgementDB{DBTX: f.pool})).User("user-1")
	target := ArtifactRef{Namespace: "docs", Name: "committed-repair"}
	if _, err := store.Write(f.ctx, target, f.payload, nil); !errors.Is(err, errLostAcknowledgement) {
		t.Fatalf("lost repair ack: %v", err)
	}
	for _, ref := range []ArtifactRef{f.first.Ref, target} {
		read, err := f.store.Read(f.ctx, ref)
		if err != nil || string(read.Payload.Data) != string(f.payload.Data) {
			t.Fatalf("committed repair lost bytes: %v", err)
		}
	}
	if _, err := f.store.Write(f.ctx, target, f.payload, nil); err == nil {
		t.Fatal("retry bypassed CAS")
	}
}
