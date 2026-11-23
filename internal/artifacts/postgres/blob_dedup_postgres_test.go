package postgres_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestFilesystemDedupRepairsMissingContent(t *testing.T) {
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
	store, _ := NewService(artifactStorage.Repository(pool)).User("user-1")
	payload := Payload{MediaType: "text/plain", Data: []byte("deduplicated payload")}
	first, err := store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "first"}, payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	beforeRepair, err := store.Metadata(ctx, first.Ref)
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
	if _, err := store.Read(ctx, first.Ref); !errors.Is(err, blob.ErrMissing) {
		t.Fatalf("expected missing content before repair: %v", err)
	}
	second, err := store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "second"}, payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	afterRepair, err := store.Metadata(ctx, first.Ref)
	if err != nil || afterRepair.StorageTimestamps.BlobCreatedAt != beforeRepair.StorageTimestamps.BlobCreatedAt {
		t.Fatalf("repair changed blob registration date: %v", err)
	}
	for _, ref := range []ArtifactRef{second.Ref, first.Ref} {
		read, err := store.Read(ctx, ref)
		if err != nil || string(read.Payload.Data) != string(payload.Data) {
			t.Fatalf("successful deduplicated write left unreadable revision: %v", err)
		}
	}
	report, err := artifactpostgres.CleanupFilesystemBlobs(ctx, pool, path, false)
	if err != nil || report.Referenced != 1 || report.Missing != 0 || report.Orphans != 0 {
		t.Fatalf("repair registry/files: %+v %v", report, err)
	}
}

type dedupFixture struct {
	storage *artifactpostgres.Storage
	ctx     context.Context
	pool    *pgxpool.Pool
	path    string
	files   *filesystem.Store
	store   ScopedStore
	payload Payload
	first   WriteResult
	key     string
}

func newDedupFixture(t *testing.T) dedupFixture {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	pool := isolatedArtifactPool(t, ctx)
	path := t.TempDir()
	files, err := filesystem.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = files.Close() })
	artifactRuntime = blob.NewRuntime(files, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	store, _ := NewService(artifactStorage.Repository(pool)).User("user-1")
	payload := Payload{MediaType: "text/plain", Data: []byte("deduplicated payload")}
	first, err := store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "first"}, payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	var key string
	if err := pool.QueryRow(ctx, `SELECT object_key FROM artifact_blobs`).Scan(&key); err != nil {
		t.Fatal(err)
	}
	return dedupFixture{artifactStorage, ctx, pool, path, files, store, payload, first, key}
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
	report, err := artifactpostgres.CleanupFilesystemBlobs(f.ctx, f.pool, f.path, false)
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
	written, err := f.storage.Repository(f.pool).CreateFrozen(f.ctx, mustProjectScope(t, "dedup-audit"),
		ArtifactRef{Namespace: "audit-test", Name: "repaired"}, f.payload)
	if err != nil {
		t.Fatal(err)
	}
	project, _ := NewService(f.storage.Repository(f.pool)).Project("dedup-audit")
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
		store, _ := NewService(f.storage.Repository(tx)).User("user-1")
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
	if _, err := f.store.Read(f.ctx, f.first.Ref); !errors.Is(err, blob.ErrMissing) {
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
	if !errors.Is(err, blob.ErrIntegrity) {
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
	prepared, err := f.storage.PreparePayload(f.ctx, f.pool, payload)
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
	if !errors.Is(err, blob.ErrMissing) {
		t.Fatalf("missing candidate published: %v", err)
	}
	var revisions int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_binding_revisions`).Scan(&revisions); err != nil || revisions != 1 {
		t.Fatalf("unexpected revision: %d %v", revisions, err)
	}
}

type recordingBlobRead struct {
	blob.Storage
	mu    sync.Mutex
	reads []string
}

func (s *recordingBlobRead) Read(ctx context.Context, object blob.Object) ([]byte, error) {
	s.mu.Lock()
	s.reads = append(s.reads, object.Key)
	s.mu.Unlock()
	return s.Storage.Read(ctx, object)
}

func TestFilesystemDedupReadsOnlyTheExistingObject(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	f := newDedupFixture(t)
	recording := &recordingBlobRead{Storage: f.files}
	ctx := f.ctx
	artifactRuntime = blob.NewRuntime(recording, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	unique := Payload{MediaType: "text/plain", Data: []byte("unique payload")}
	if _, err := routingUser(t, artifactStorage, f.pool, "user-1").Write(ctx, ArtifactRef{Namespace: "docs", Name: "unique"}, unique, nil); err != nil {
		t.Fatal(err)
	}
	if len(recording.reads) != 0 {
		t.Fatalf("write without a deduplication match read %v", recording.reads)
	}
	prepared, err := artifactStorage.PreparePayload(ctx, f.pool, f.payload)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := routingUser(t, artifactStorage, f.pool, "user-1").Write(ctx, ArtifactRef{Namespace: "docs", Name: "second"}, prepared, nil); err != nil {
		t.Fatal(err)
	}
	if len(recording.reads) != 1 || recording.reads[0] != f.key {
		t.Fatalf("deduplicated write read %v, want only the existing key %s", recording.reads, f.key)
	}
}

func TestFilesystemDedupRejectsTruncatedPreparedCandidate(t *testing.T) {
	f := newDedupFixture(t)
	prepared, err := f.storage.PreparePayload(f.ctx, f.pool, Payload{MediaType: "text/plain", Data: []byte("prepared candidate")})
	if err != nil {
		t.Fatal(err)
	}
	err = filepath.WalkDir(f.path, func(path string, entry os.DirEntry, err error) error {
		if err != nil || entry.IsDir() || path == filepath.Join(f.path, f.key) {
			return err
		}
		return os.Truncate(path, 1)
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.store.Write(f.ctx, ArtifactRef{Namespace: "docs", Name: "not-published"}, prepared, nil)
	if !errors.Is(err, blob.ErrIntegrity) {
		t.Fatalf("truncated candidate error = %v", err)
	}
}

type pausedBlobRead struct {
	blob.Storage
	key     string
	entered chan struct{}
	resume  chan struct{}
}

func (s pausedBlobRead) Read(ctx context.Context, object blob.Object) ([]byte, error) {
	data, err := s.Storage.Read(ctx, object)
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
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	f := newDedupFixture(t)
	paused := pausedBlobRead{Storage: f.files, key: f.key, entered: make(chan struct{}, 1), resume: make(chan struct{})}
	ctx := f.ctx
	artifactRuntime = blob.NewRuntime(paused, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	finished := make(chan error, 1)
	go func() {
		_, err := routingUser(t, artifactStorage, f.pool, "user-1").Write(ctx, ArtifactRef{Namespace: "docs", Name: "delayed"}, f.payload, nil)
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
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	f := newDedupFixture(t)
	if err := os.Remove(filepath.Join(f.path, f.key)); err != nil {
		t.Fatal(err)
	}
	paused := pausedBlobRead{Storage: f.files, key: f.key, entered: make(chan struct{}, 2), resume: make(chan struct{})}
	ctx := f.ctx
	artifactRuntime = blob.NewRuntime(paused, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	var release sync.Once
	var writers sync.WaitGroup
	finished := make(chan error, 2)
	defer func() { release.Do(func() { close(paused.resume) }); writers.Wait() }()
	for _, name := range []string{"second", "third"} {
		writers.Add(1)
		go func() {
			defer writers.Done()
			_, err := routingUser(t, artifactStorage, f.pool, "user-1").Write(ctx, ArtifactRef{Namespace: "docs", Name: name}, f.payload, nil)
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
	report, err := artifactpostgres.CleanupFilesystemBlobs(f.ctx, f.pool, f.path, false)
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
	store, _ := NewService(f.storage.Repository(lostRepairAcknowledgementDB{DBTX: f.pool})).User("user-1")
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
