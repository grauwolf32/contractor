package artifacts_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The production read must be a registry-only projection, even for filesystem
// payloads. Calling any payload operation makes this test fail immediately.
type timestampNoBlobs struct{}

func (timestampNoBlobs) Store(context.Context, []byte) (BlobObject, error) {
	panic("metadata stored payload")
}
func (timestampNoBlobs) Read(context.Context, BlobObject) ([]byte, error) {
	panic("metadata read payload")
}
func (timestampNoBlobs) Check(context.Context, BlobObject) error  { panic("metadata checked payload") }
func (timestampNoBlobs) Delete(context.Context, BlobObject) error { panic("metadata deleted payload") }

type timestampQueries struct {
	postgres.DBTX
	queries int
}

func (p *timestampQueries) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	p.queries++
	return p.DBTX.Query(ctx, sql, args...)
}
func (p *timestampQueries) QueryRow(ctx context.Context, sql string, args ...any) pgx.Row {
	p.queries++
	return p.DBTX.QueryRow(ctx, sql, args...)
}

func TestPostgresArtifactStorageDatesExactForkCASAndPurge(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx = testBlobContext(t, ctx, pool)
	service := NewService(NewPostgresRepository(pool))
	user, _ := service.User("user-1")
	payload := Payload{MediaType: "application/json", Data: []byte(`{"storageTimestamps":{"blobCreatedAt":"invented"}}`)}
	first, err := user.Write(ctx, ArtifactRef{Namespace: "docs", Name: "source"}, payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	source, err := user.Metadata(ctx, first.Ref)
	if err != nil {
		t.Fatal(err)
	}
	createArtifactRun(t, ctx, pool, "timestamp-fork", false)
	fork, err := artifactstest.ForkInput(ctx, service, "user-1", first.Ref, "timestamp-fork", "source")
	if err != nil {
		t.Fatal(err)
	}
	run, _ := service.Run("timestamp-fork")
	target, err := run.Metadata(ctx, fork.TargetRef)
	if err != nil {
		t.Fatal(err)
	}
	if !target.StorageTimestamps.BlobCreatedAt.Equal(source.StorageTimestamps.BlobCreatedAt) || !target.StorageTimestamps.VersionCreatedAt.Equal(source.StorageTimestamps.VersionCreatedAt) {
		t.Fatal("fork changed shared storage creation")
	}
	duplicate, err := user.Write(ctx, ArtifactRef{Namespace: "docs", Name: "duplicate"}, payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	dup, err := user.Metadata(ctx, duplicate.Ref)
	if err != nil {
		t.Fatal(err)
	}
	if !dup.StorageTimestamps.BlobCreatedAt.Equal(source.StorageTimestamps.BlobCreatedAt) || dup.ContentCacheKey == source.ContentCacheKey {
		t.Fatal("dedup confused blob and version identity")
	}
	next, err := user.Write(ctx, ArtifactRef{Namespace: "docs", Name: "source"}, Payload{MediaType: "text/plain", Data: []byte("new version")}, first.Ref.Revision)
	if err != nil {
		t.Fatal(err)
	}
	current, err := user.Metadata(ctx, next.Ref)
	if err != nil {
		t.Fatal(err)
	}
	historical, err := user.Metadata(ctx, first.Ref)
	if err != nil {
		t.Fatal(err)
	}
	if !historical.CreatedAt.Equal(first.RevisionCreatedAt) || historical.StorageTimestamps.VersionCreatedAt != source.StorageTimestamps.VersionCreatedAt || historical.StorageTimestamps.BindingUpdatedAt != current.StorageTimestamps.BindingUpdatedAt {
		t.Fatal("historical version did not retain exact dates and current binding date")
	}
	if _, err := user.Write(ctx, ArtifactRef{Namespace: "docs", Name: "source"}, payload, first.Ref.Revision); !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("stale CAS=%v", err)
	}
	after, err := user.Metadata(ctx, next.Ref)
	if err != nil || after.StorageTimestamps != current.StorageTimestamps {
		t.Fatalf("rejected CAS changed dates: %v", err)
	}
	scope, _ := RunScope("timestamp-fork")
	if err := NewPostgresRepository(pool).Freeze(ctx, scope, "inputs"); err != nil {
		t.Fatal(err)
	}
	frozen, err := run.Metadata(ctx, fork.TargetRef)
	if err != nil || !frozen.Frozen {
		t.Fatalf("freeze: %v", err)
	}
	if !frozen.Ref.SameExact(target.Ref) || frozen.StorageTimestamps.BlobCreatedAt != target.StorageTimestamps.BlobCreatedAt || frozen.StorageTimestamps.VersionCreatedAt != target.StorageTimestamps.VersionCreatedAt {
		t.Fatal("freeze changed immutable content dates")
	}
	// A Project output survives purge of its source Run with original dates.
	project, outputRef := storageReviewProject(t, ctx, pool)
	published, err := publishOutput(ctx, service, "review-run", project.ProjectID, "result", outputRef)
	if err != nil {
		t.Fatal(err)
	}
	projectStore, _ := service.Project(project.ProjectID)
	retained, err := projectStore.Metadata(ctx, published.TargetRef)
	if err != nil {
		t.Fatal(err)
	}
	if err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		p, err := NewPostgresPurger(tx)
		if err != nil {
			return err
		}
		return p.PurgeRun(ctx, "review-run")
	}); err != nil {
		t.Fatal(err)
	}
	surviving, err := projectStore.Metadata(ctx, published.TargetRef)
	if err != nil || surviving.StorageTimestamps != retained.StorageTimestamps {
		t.Fatalf("surviving fork lost dates: %v", err)
	}
	assertStoredTimestampFacts(t, ctx, pool, ScopeProject, project.ProjectID, published.TargetRef, surviving)
	// Creation guards reject changes without blocking the mutable scope counter.
	for _, sql := range []string{`UPDATE artifact_scopes SET created_at=created_at+interval '1 second'`, `UPDATE artifact_blobs SET created_at=created_at+interval '1 second'`, `UPDATE artifact_versions SET created_at=created_at+interval '1 second'`} {
		if _, err := pool.Exec(ctx, sql); postgres.Classify(err) != postgres.ClassRejected {
			t.Fatalf("creation guard=%v", err)
		}
	}
	if _, err := pool.Exec(ctx, `UPDATE artifact_scopes SET revision_count=revision_count WHERE scope_kind='run'`); err != nil {
		t.Fatal(err)
	}
}

func assertStoredTimestampFacts(t *testing.T, ctx context.Context, pool *pgxpool.Pool, scopeKind ScopeKind, scopeID string, ref ArtifactRef, got Metadata) {
	t.Helper()
	var expected StorageTimestamps
	var revisionAt time.Time
	err := pool.QueryRow(ctx, `SELECT scope.created_at,blob.created_at,version.created_at,binding.updated_at,revision.created_at
 FROM artifact_scopes scope JOIN artifact_bindings binding USING(scope_kind,scope_id)
 JOIN artifact_binding_revisions revision USING(scope_kind,scope_id,namespace,name)
 JOIN artifact_versions version USING(version_id) JOIN artifact_blobs blob ON blob.sha256=version.blob_sha256
 WHERE scope.scope_kind=$1 AND scope.scope_id=$2 AND binding.namespace=$3 AND binding.name=$4 AND revision.revision=$5`, scopeKind, scopeID, ref.Namespace, ref.Name, ref.Revision).Scan(&expected.ScopeCreatedAt, &expected.BlobCreatedAt, &expected.VersionCreatedAt, &expected.BindingUpdatedAt, &revisionAt)
	if err != nil {
		t.Fatal(err)
	}
	for _, pair := range [][2]time.Time{{expected.ScopeCreatedAt, got.StorageTimestamps.ScopeCreatedAt}, {expected.BlobCreatedAt, got.StorageTimestamps.BlobCreatedAt}, {expected.VersionCreatedAt, got.StorageTimestamps.VersionCreatedAt}, {expected.BindingUpdatedAt, got.StorageTimestamps.BindingUpdatedAt}, {revisionAt, got.CreatedAt}} {
		if !pair[0].Equal(pair[1]) || pair[1].Location() != time.UTC {
			t.Fatalf("stored date mismatch: %v", pair)
		}
	}
}

func TestPostgresArtifactStorageDatesBoundedPages(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx = testBlobContext(t, ctx, pool)
	store, _ := NewService(NewPostgresRepository(pool)).User("timestamp-pages")
	seed, err := store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "seed"}, Payload{MediaType: "text/plain", Data: []byte("seed")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := store.Metadata(ctx, seed.Ref)
	if err != nil {
		t.Fatal(err)
	}
	// Shared immutable payloads make a realistic metadata population cheap; ties
	// in revision creation are intentional to exercise the secondary cursor key.
	seedTx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer seedTx.Rollback(context.Background())
	for _, sql := range []string{
		`INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,retained_revisions,retained_bytes) SELECT 'user','timestamp-pages','docs','item-'||lpad(i::text,4,'0'),'r1',1,4 FROM generate_series(1,1000)i`,
		`INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id,created_at) SELECT 'user','timestamp-pages','docs','item-'||lpad(i::text,4,'0'),'r1',$1,'2026-01-01T00:00:00Z' FROM generate_series(1,1000)i`,
		`INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id,created_at) SELECT 'user','timestamp-pages','docs','seed','history-'||lpad(i::text,3,'0'),$1,'2026-01-01T00:00:00Z' FROM generate_series(1,250)i`,
	} {
		args := []any{}
		if strings.Contains(sql, "$1") {
			args = append(args, metadata.ContentCacheKey)
		}
		if _, err := seedTx.Exec(ctx, sql, args...); err != nil {
			t.Fatal(err)
		}
	}
	if err := seedTx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(context.Background())
	if _, err := tx.Exec(ctx, `SET LOCAL TIME ZONE 'Pacific/Honolulu'`); err != nil {
		t.Fatal(err)
	}
	reads := &timestampQueries{DBTX: tx}
	repository := NewPostgresRepository(reads)
	scope, _ := UserScope("timestamp-pages")
	readCtx := WithBlobRuntime(ctx, NewBlobRuntime(timestampNoBlobs{}, nil))
	for _, size := range []int{7, 200} {
		query := BindingPageQuery{Limit: size}
		seen := map[string]bool{}
		for {
			before := reads.queries
			page, err := repository.ListMetadata(readCtx, scope, query)
			if err != nil {
				t.Fatal(err)
			}
			if reads.queries != before+1 || len(page) > size {
				t.Fatal("metadata page exceeded query/row bound")
			}
			for _, item := range page {
				if seen[item.Ref.Name] {
					t.Fatal("duplicate binding")
				}
				seen[item.Ref.Name] = true
				encoded, _ := json.Marshal(item.StorageTimestamps)
				if len(encoded) > 512 || item.StorageTimestamps.ScopeCreatedAt.Location() != time.UTC {
					t.Fatal("invalid timestamp projection")
				}
			}
			if len(page) < size {
				break
			}
			last := page[len(page)-1]
			query.AfterNamespace = last.Ref.Namespace
			query.AfterName = last.Ref.Name
		}
		if len(seen) != 1001 {
			t.Fatalf("read %d bindings", len(seen))
		}
		versions := VersionPageQuery{Limit: size}
		seen = map[string]bool{}
		for {
			before := reads.queries
			page, err := repository.ListVersions(readCtx, scope, ArtifactRef{Namespace: seed.Ref.Namespace, Name: seed.Ref.Name}, versions)
			if err != nil {
				t.Fatal(err)
			}
			if reads.queries != before+1 || len(page) > size {
				t.Fatal("version page exceeded bound")
			}
			for _, item := range page {
				if seen[*item.Ref.Revision] {
					t.Fatal("duplicate revision")
				}
				seen[*item.Ref.Revision] = true
			}
			if len(page) < size {
				break
			}
			last := page[len(page)-1]
			versions.BeforeCreatedAt = &last.CreatedAt
			versions.BeforeRevision = *last.Ref.Revision
		}
		if len(seen) != 251 {
			t.Fatalf("read %d revisions", len(seen))
		}
	}
}

func TestPostgresArtifactStorageDatesRejectCorruption(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	if _, err := pool.Exec(ctx, `INSERT INTO artifact_scopes(scope_kind,scope_id,created_at) VALUES('user','corrupt-date','0001-01-01T00:00:00Z')`); err != nil {
		t.Fatal(err)
	}
	store, _ := NewService(NewPostgresRepository(pool)).User("corrupt-date")
	written, err := store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "source"}, Payload{MediaType: "text/plain", Data: []byte("valid payload")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, read := range []func() error{func() error { _, err := store.Metadata(ctx, written.Ref); return err }, func() error { _, err := store.ListMetadata(ctx, BindingPageQuery{Limit: 10}); return err }, func() error {
		_, err := store.ListVersions(ctx, ArtifactRef{Namespace: written.Ref.Namespace, Name: written.Ref.Name}, VersionPageQuery{Limit: 10})
		return err
	}} {
		if err := read(); !errors.Is(err, ErrArtifactIntegrity) {
			t.Fatalf("corrupt date=%v", err)
		}
	}
	cancelled, stop := context.WithCancel(ctx)
	stop()
	if _, err := store.Metadata(cancelled, written.Ref); err == nil {
		t.Fatal("cancelled metadata read succeeded")
	}
}
