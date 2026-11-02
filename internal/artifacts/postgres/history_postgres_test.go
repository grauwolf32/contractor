package postgres_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// historyBinding names one binding in the SQL helpers of these tests.
type historyBinding struct {
	kind, id, namespace, name string
}

func (b historyBinding) ref() ArtifactRef { return ArtifactRef{Namespace: b.namespace, Name: b.name} }

// Every binding of every scope keeps its own retained history without
// registration: once a write would exceed the limit, the oldest revisions go
// in the same transaction, whatever the name or media type, and the content
// they alone referenced goes with them while shared content stays. Exact
// reads of a removed revision report not found, never the current content.
func TestPostgresHistoryRetainsTheNewestRevisionsOfEachBinding(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx, _ = purgeBlobContext(t, ctx, pool)
	ctx = artifactpostgres.WithHistoryPolicy(ctx, artifactpostgres.NewTestHistoryPolicy(3, 1<<30))
	createHistoryProject(t, ctx, pool, "history-project")
	createArtifactRun(t, ctx, pool, "history-run", true)
	service := NewService(artifactpostgres.NewRepository(pool))
	user, _ := service.User("user-1")
	project, _ := service.Project("history-project")
	run, _ := service.Run("history-run")
	shared := Payload{MediaType: "text/plain", Data: []byte("shared by another binding")}
	if _, err := user.Write(ctx, ArtifactRef{Namespace: "keep", Name: "shared"}, shared, nil); err != nil {
		t.Fatal(err)
	}
	for _, scope := range []struct {
		store ScopedStore
		kind  string
		id    string
	}{{user, "user", "user-1"}, {project, "project", "history-project"}, {run, "run", "history-run"}} {
		for _, target := range []struct {
			binding   historyBinding
			mediaType string
		}{
			{historyBinding{scope.kind, scope.id, "notes", "plain"}, "text/plain"},
			{historyBinding{scope.kind, scope.id, "data", "document"}, "application/json"},
		} {
			var written []string
			var revision *string
			for version := range 5 {
				payload := Payload{MediaType: target.mediaType, Data: fmt.Appendf(nil, `"%s %s %d"`, scope.kind, target.binding.name, version)}
				if version == 0 {
					payload = Payload{MediaType: target.mediaType, Data: shared.Data}
					if target.mediaType != "text/plain" {
						payload.Data = []byte(`"shared"`)
					}
				}
				result, err := scope.store.Write(ctx, target.binding.ref(), payload, revision)
				if err != nil {
					t.Fatalf("%s write %d: %v", target.binding, version, err)
				}
				revision = result.Ref.Revision
				written = append(written, *revision)
				if revisions, _ := historyCounters(t, ctx, pool, target.binding); revisions != int64(min(version+1, 3)) {
					t.Fatalf("%s retains %d revisions after write %d", target.binding, revisions, version)
				}
			}
			if got := historyRevisions(t, ctx, pool, target.binding); !slices.Equal(got, written[2:]) {
				t.Fatalf("%s retains %v, want the newest three of %v", target.binding, got, written)
			}
			assertHistoryCounters(t, ctx, pool, target.binding)
			removed := target.binding.ref()
			removed.Revision = &written[1]
			if _, err := scope.store.Read(ctx, removed); !errors.Is(err, ErrArtifactNotFound) {
				t.Fatalf("read removed revision of %s = %v, want not found", target.binding, err)
			}
			current, err := scope.store.Read(ctx, target.binding.ref())
			if err != nil || *current.Ref.Revision != written[4] {
				t.Fatalf("current of %s = %v, %v", target.binding, current.Ref, err)
			}
		}
	}
	// The shared content survives its removed revisions, and content only a
	// removed revision used is collected with its version.
	var sharedBlobs, orphanVersions int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_blobs WHERE sha256 = sha256($1::bytea)`,
		shared.Data).Scan(&sharedBlobs); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_versions AS version WHERE NOT EXISTS (
SELECT 1 FROM artifact_binding_revisions AS revision WHERE revision.version_id = version.version_id)`).
		Scan(&orphanVersions); err != nil {
		t.Fatal(err)
	}
	if sharedBlobs != 1 || orphanVersions != 0 {
		t.Fatalf("shared blobs = %d, unreferenced versions = %d; want 1 and 0", sharedBlobs, orphanVersions)
	}
	var blobs, referenced int
	if err := pool.QueryRow(ctx, `SELECT count(*), count(*) FILTER (WHERE EXISTS (
SELECT 1 FROM artifact_versions AS version WHERE version.blob_sha256 = blob.sha256)) FROM artifact_blobs AS blob`).
		Scan(&blobs, &referenced); err != nil {
		t.Fatal(err)
	}
	if blobs != referenced {
		t.Fatalf("%d of %d blobs are unreferenced after retention", blobs-referenced, blobs)
	}
}

// The byte limit counts every retained revision's full size, also when two
// revisions share content, and the limit applies to each binding on its own:
// two bindings together retain more than one binding may.
func TestPostgresHistoryByteLimitAppliesToEachBinding(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx, _ = purgeBlobContext(t, ctx, pool)
	ctx = artifactpostgres.WithHistoryPolicy(ctx, artifactpostgres.NewTestHistoryPolicy(100, 25))
	user, _ := NewService(artifactpostgres.NewRepository(pool)).User("user-1")
	for _, name := range []string{"first", "second"} {
		binding := historyBinding{"user", "user-1", "bytes", name}
		var revision *string
		var written []string
		for _, data := range []string{"0123456789", "0123456789", "abcdefghij", "klmno"} {
			result, err := user.Write(ctx, binding.ref(), Payload{MediaType: "text/plain", Data: []byte(data)}, revision)
			if err != nil {
				t.Fatal(err)
			}
			revision = result.Ref.Revision
			written = append(written, *revision)
		}
		// 10+10+10 exceeds 25, so the third write removed the first; 10+10+5
		// fits.
		if got := historyRevisions(t, ctx, pool, binding); !slices.Equal(got, written[1:]) {
			t.Fatalf("%s retains %v, want %v", name, got, written[1:])
		}
		if revisions, bytes := historyCounters(t, ctx, pool, binding); revisions != 3 || bytes != 25 {
			t.Fatalf("%s counters = %d revisions, %d bytes; want 3 and 25", name, revisions, bytes)
		}
	}
	// A single revision above the byte limit is protected history: the new
	// current alone exceeds it.
	binding := historyBinding{"user", "user-1", "bytes", "first"}
	current, err := user.Read(ctx, binding.ref())
	if err != nil {
		t.Fatal(err)
	}
	_, err = user.Write(ctx, binding.ref(), Payload{MediaType: "text/plain", Data: []byte(strings.Repeat("x", 26))}, current.Ref.Revision)
	var protected *HistoryProtectedError
	if !errors.As(err, &protected) || !errors.Is(err, ErrHistoryProtected) || protected.RetainedBytes != 26 {
		t.Fatalf("oversized write = %v, want protected history of 26 bytes", err)
	}
	if revisions, bytes := historyCounters(t, ctx, pool, binding); revisions != 3 || bytes != 25 {
		t.Fatalf("a protected write changed the counters to %d revisions, %d bytes", revisions, bytes)
	}
}

// Exact holds keep revisions: retention removes the oldest unprotected
// revision, which may be the former current, never a held one, and refuses a
// write that only held revisions could make room for. Releasing a hold puts
// its revision back first in line, ahead of every younger revision.
func TestPostgresHistoryPreservesHoldsAndReseeksAReleasedHold(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx, _ = purgeBlobContext(t, ctx, pool)
	ctx = artifactpostgres.WithHistoryPolicy(ctx, artifactpostgres.NewTestHistoryPolicy(3, 1<<30))
	service := NewService(artifactpostgres.NewRepository(pool))
	user, _ := service.User("user-1")
	binding := historyBinding{"user", "user-1", "notes", "held"}
	var revisions []string
	write := func(data string) error {
		var expected *string
		if len(revisions) > 0 {
			current, err := user.Read(ctx, binding.ref())
			if err != nil {
				return err
			}
			expected = current.Ref.Revision
		}
		result, err := user.Write(ctx, binding.ref(), Payload{MediaType: "text/plain", Data: []byte(data)}, expected)
		if err == nil {
			revisions = append(revisions, *result.Ref.Revision)
		}
		return err
	}
	hold := func(runID string, revision string) {
		t.Helper()
		createArtifactRun(t, ctx, pool, runID, true)
		source := binding.ref()
		source.Revision = &revision
		if _, err := artifactstest.ForkInput(ctx, service, "user-1", source, runID, "held"); err != nil {
			t.Fatal(err)
		}
	}
	for _, data := range []string{"r1", "r2"} {
		if err := write(data); err != nil {
			t.Fatal(err)
		}
	}
	hold("hold-r1", revisions[0])
	if err := write("r3"); err != nil {
		t.Fatal(err)
	}
	if err := write("r4"); err != nil { // removes r2, the oldest unprotected
		t.Fatal(err)
	}
	if got := historyRevisions(t, ctx, pool, binding); !slices.Equal(got, []string{revisions[0], revisions[2], revisions[3]}) {
		t.Fatalf("retained %v after r4", got)
	}
	hold("hold-r3", revisions[2])
	if err := write("r5"); err != nil { // removes r4, the former current
		t.Fatal(err)
	}
	if got := historyRevisions(t, ctx, pool, binding); !slices.Equal(got, []string{revisions[0], revisions[2], revisions[4]}) {
		t.Fatalf("retained %v after r5, want the former current removed", got)
	}
	hold("hold-r5", revisions[4])
	before := historyRevisions(t, ctx, pool, binding)
	err := write("r6")
	var protected *HistoryProtectedError
	if !errors.As(err, &protected) || protected.RetainedRevisions != 4 || protected.MaxRevisions != 3 {
		t.Fatalf("write over held history = %v, want protected history of 4 revisions", err)
	}
	if got := historyRevisions(t, ctx, pool, binding); !slices.Equal(got, before) {
		t.Fatalf("a protected write changed the history to %v from %v", got, before)
	}
	assertHistoryCounters(t, ctx, pool, binding)

	// Purging the Run that held r1 releases it; the next write removes r1,
	// the oldest, although r3 and r5 are younger and still held.
	purgeHistoryRun(t, ctx, pool, "hold-r1")
	if count := historyHoldCount(t, ctx, pool, binding, revisions[0]); count != 0 {
		t.Fatalf("released r1 keeps %d holds", count)
	}
	if err := write("r6"); err != nil {
		t.Fatal(err)
	}
	if got := historyRevisions(t, ctx, pool, binding); !slices.Equal(got, []string{revisions[2], revisions[4], revisions[5]}) {
		t.Fatalf("retained %v after the release, want r1 removed first", got)
	}
	assertHistoryCounters(t, ctx, pool, binding)
	// A fork of the removed revision reports it missing rather than forking
	// the current revision.
	removed := binding.ref()
	removed.Revision = &revisions[0]
	createArtifactRun(t, ctx, pool, "late-run", true)
	if _, err := artifactstest.ForkInput(ctx, service, "user-1", removed, "late-run", "late"); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("fork of a removed revision = %v, want not found", err)
	}
}

// One transaction examines at most 64 history candidates and removes at most
// 64 revisions. A write that needs more commits one bounded maintenance pass
// of its own binding and reports it as pending; later writes continue from
// the oldest unprotected revision. Thousands of unrelated bindings and held
// revisions are never examined, and a hold released behind the removed
// revisions is removed before any younger revision.
func TestPostgresHistoryMaintenanceIsBoundedPerTransaction(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx, _ = purgeBlobContext(t, ctx, pool)
	ctx = artifactpostgres.WithHistoryPolicy(ctx, artifactpostgres.NewTestHistoryPolicy(20, 1<<30))
	service := NewService(artifactpostgres.NewRepository(pool))
	user, _ := service.User("user-1")
	seedUnrelatedBindings(t, ctx, pool, 2000)
	binding := historyBinding{"user", "user-1", "notes", "long"}
	seeded := seedHistory(t, ctx, pool, binding, 200)
	// Hold the five oldest revisions through a Run input each.
	for index := range 5 {
		runID := fmt.Sprintf("hold-%d", index)
		createArtifactRun(t, ctx, pool, runID, true)
		source := binding.ref()
		source.Revision = &seeded[index]
		if _, err := artifactstest.ForkInput(ctx, service, "user-1", source, runID, "held"); err != nil {
			t.Fatal(err)
		}
	}
	current := seeded[len(seeded)-1]
	write := func() error {
		result, err := user.Write(ctx, binding.ref(), Payload{MediaType: "text/plain", Data: []byte("next " + current)}, &current)
		if err == nil {
			current = *result.Ref.Revision
		}
		return err
	}
	// 200 retained + 1 needs 181 removals: two passes of 64 each.
	for pass, wantRemaining := range []int{200 - 64, 200 - 128} {
		err := write()
		var pending *HistoryMaintenanceError
		if !errors.As(err, &pending) || !pending.Maintained || pending.Removed != 64 {
			t.Fatalf("pass %d = %v, want 64 revisions removed and the write pending", pass, err)
		}
		retained := historyRevisions(t, ctx, pool, binding)
		if len(retained) != wantRemaining {
			t.Fatalf("pass %d left %d revisions, want %d", pass, len(retained), wantRemaining)
		}
		// The held five and the youngest revisions survive; removal went in
		// creation order.
		if !slices.Equal(retained[:5], seeded[:5]) || retained[5] != seeded[len(seeded)-wantRemaining+5] {
			t.Fatalf("pass %d retained %v...", pass, retained[:6])
		}
		assertHistoryCounters(t, ctx, pool, binding)
	}
	// Release two holds; the next write removes them before the younger
	// unprotected revisions it needs to remove, and now fits one transaction.
	purgeHistoryRun(t, ctx, pool, "hold-0")
	purgeHistoryRun(t, ctx, pool, "hold-1")
	previous := current
	if err := write(); err != nil {
		t.Fatalf("write after maintenance: %v", err)
	}
	want := append(append(slices.Clone(seeded[2:5]), seeded[184:]...), current)
	if got := historyRevisions(t, ctx, pool, binding); !slices.Equal(got, want) || previous != seeded[199] {
		t.Fatalf("retained %v after the release, want %v", got, want)
	}
	assertHistoryCounters(t, ctx, pool, binding)
	var unrelated int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_binding_revisions WHERE namespace = 'unrelated'`).
		Scan(&unrelated); err != nil {
		t.Fatal(err)
	}
	if unrelated != 2000 {
		t.Fatalf("unrelated bindings retain %d revisions, want 2000", unrelated)
	}
}

// The 64-removal bound spans every binding a caller's transaction writes: a
// second write that would exceed it is pending without committing anything,
// and once the caller has rolled back, MaintainHistory commits one pass for
// that binding.
func TestPostgresHistoryBoundSpansTheCallersTransaction(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx, _ = purgeBlobContext(t, ctx, pool)
	ctx = artifactpostgres.WithHistoryPolicy(ctx, artifactpostgres.NewTestHistoryPolicy(10, 1<<30))
	first := historyBinding{"user", "user-1", "notes", "first"}
	second := historyBinding{"user", "user-1", "notes", "second"}
	firstSeeded := seedHistory(t, ctx, pool, first, 50)
	secondSeeded := seedHistory(t, ctx, pool, second, 50)
	var writeErr error
	err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		user, _ := NewService(artifactpostgres.NewRepository(tx)).User("user-1")
		if _, err := user.Write(ctx, first.ref(), Payload{MediaType: "text/plain", Data: []byte("first")}, &firstSeeded[49]); err != nil {
			return err
		}
		_, writeErr = user.Write(ctx, second.ref(), Payload{MediaType: "text/plain", Data: []byte("second")}, &secondSeeded[49])
		return writeErr
	})
	var pending *HistoryMaintenanceError
	if !errors.As(err, &pending) || pending.Maintained {
		t.Fatalf("second write in one transaction = %v, want pending without maintenance", err)
	}
	if len(historyRevisions(t, ctx, pool, first)) != 50 || len(historyRevisions(t, ctx, pool, second)) != 50 {
		t.Fatal("the rolled back transaction removed history")
	}
	err = artifactpostgres.NewRepository(pool).MaintainHistory(ctx, err)
	if !errors.As(err, &pending) || !pending.Maintained || pending.Removed != 41 {
		t.Fatalf("maintenance after rollback = %v, want 41 removals", err)
	}
	if len(historyRevisions(t, ctx, pool, first)) != 50 || len(historyRevisions(t, ctx, pool, second)) != 9 {
		t.Fatal("maintenance changed another binding or kept too much")
	}
	assertHistoryCounters(t, ctx, pool, second)
	// A repository bound to a transaction cannot commit a pass.
	err = postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		return artifactpostgres.NewRepository(tx).MaintainHistory(ctx, &HistoryMaintenanceError{
			Scope: userScope(t, "user-1"), Ref: second.ref(), Size: 1,
		})
	})
	if err == nil || !errors.Is(err, ErrHistoryMaintenancePending) {
		t.Fatalf("maintenance inside a transaction = %v", err)
	}
}

// One pass removes at most 64 MiB of encoded content; the first removal of a
// pass always goes, so a pass makes progress whatever the sizes.
func TestPostgresHistoryPassReclaimsAtMost64MiB(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx, objects := purgeBlobContext(t, ctx, pool)
	if objects == "" {
		files, err := filesystem.Open(ctx, t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = files.Close() })
		ctx = blob.WithRuntime(ctx, blob.NewRuntime(files, nil))
	}
	ctx = artifactpostgres.WithHistoryPolicy(ctx, artifactpostgres.NewTestHistoryPolicy(2, 1<<40))
	binding := historyBinding{"user", "user-1", "files", "large"}
	// Four 40-MiB revisions recorded as filesystem content whose objects are
	// already gone: their unlink after commit is a logged no-op.
	seeded := seedLargeHistory(t, ctx, pool, binding, 4, 40<<20)
	user, _ := NewService(artifactpostgres.NewRepository(pool)).User("user-1")
	_, err := user.Write(ctx, binding.ref(), Payload{MediaType: "text/plain", Data: []byte("small")}, &seeded[3])
	var pending *HistoryMaintenanceError
	if !errors.As(err, &pending) || pending.Removed != 1 {
		t.Fatalf("write needing three 40-MiB removals = %v, want one removal per pass", err)
	}
	if got := historyRevisions(t, ctx, pool, binding); !slices.Equal(got, seeded[1:]) {
		t.Fatalf("retained %v after one pass", got)
	}
	var blobs int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_blobs WHERE backend = 'filesystem'`).Scan(&blobs); err != nil {
		t.Fatal(err)
	}
	if blobs != 3 {
		t.Fatalf("%d filesystem blobs remain, want 3", blobs)
	}
}

// Writes that only create bindings, refused CAS writes and frozen bindings
// keep their existing outcomes and never touch history.
func TestPostgresHistoryLeavesOtherWriteOutcomesAlone(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx, _ = purgeBlobContext(t, ctx, pool)
	ctx = artifactpostgres.WithHistoryPolicy(ctx, artifactpostgres.NewTestHistoryPolicy(2, 1<<30))
	service := NewService(artifactpostgres.NewRepository(pool))
	user, _ := service.User("user-1")
	binding := historyBinding{"user", "user-1", "notes", "cas"}
	seeded := seedHistory(t, ctx, pool, binding, 5)
	if _, err := user.Write(ctx, binding.ref(), Payload{MediaType: "text/plain", Data: []byte("stale")}, &seeded[0]); !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("stale CAS = %v, want conflict", err)
	}
	if _, err := user.Write(ctx, binding.ref(), Payload{MediaType: "text/plain", Data: []byte("create")}, nil); !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("create of an existing binding = %v, want conflict", err)
	}
	missing := "missing"
	if _, err := user.Write(ctx, ArtifactRef{Namespace: "notes", Name: "absent"}, Payload{MediaType: "text/plain", Data: []byte("x")}, &missing); !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("CAS of a missing binding = %v, want conflict", err)
	}
	if len(historyRevisions(t, ctx, pool, binding)) != 5 {
		t.Fatal("a refused write removed history")
	}
	// A frozen binding over its history limit stays frozen: the write is
	// refused as frozen, and nothing is removed.
	frozen := historyBinding{"user", "user-1", "frozen", "notes"}
	frozenSeeded := seedHistory(t, ctx, pool, frozen, 5)
	if err := service.Freeze(ctx, userScope(t, "user-1"), "frozen"); err != nil {
		t.Fatal(err)
	}
	if _, err := user.Write(ctx, frozen.ref(), Payload{MediaType: "text/plain", Data: []byte("y")}, &frozenSeeded[4]); !errors.Is(err, ErrArtifactFrozen) {
		t.Fatalf("write of a frozen binding = %v, want frozen", err)
	}
	if len(historyRevisions(t, ctx, pool, frozen)) != 5 {
		t.Fatal("a frozen binding lost history")
	}
}

func createHistoryProject(t *testing.T, ctx context.Context, pool *pgxpool.Pool, projectID string) {
	t.Helper()
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "History", IdempotencyKey: "create-" + projectID,
		RequestDigest: "sha256:" + strings.Repeat("d", 64),
	}); err != nil {
		t.Fatal(err)
	}
}

// userScope returns the UserScope of id.
func userScope(t *testing.T, id string) Scope {
	t.Helper()
	scope, err := UserScope(id)
	if err != nil {
		t.Fatal(err)
	}
	return scope
}

// historyRevisions returns the retained revisions of binding in creation
// order.
func historyRevisions(t *testing.T, ctx context.Context, pool *pgxpool.Pool, binding historyBinding) []string {
	t.Helper()
	rows, err := pool.Query(ctx, `SELECT revision FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4 ORDER BY created_at, revision`,
		binding.kind, binding.id, binding.namespace, binding.name)
	if err != nil {
		t.Fatal(err)
	}
	revisions, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatal(err)
	}
	return revisions
}

func historyCounters(t *testing.T, ctx context.Context, pool *pgxpool.Pool, binding historyBinding) (int64, int64) {
	t.Helper()
	var revisions, bytes int64
	if err := pool.QueryRow(ctx, `SELECT retained_revisions, retained_bytes FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4`,
		binding.kind, binding.id, binding.namespace, binding.name).Scan(&revisions, &bytes); err != nil {
		t.Fatal(err)
	}
	return revisions, bytes
}

// assertHistoryCounters compares the binding's counters with its retained
// revisions, computed independently.
func assertHistoryCounters(t *testing.T, ctx context.Context, pool *pgxpool.Pool, binding historyBinding) {
	t.Helper()
	var revisions, bytes int64
	if err := pool.QueryRow(ctx, `SELECT count(*), COALESCE(sum(blob.size_bytes), 0)
FROM artifact_binding_revisions AS revision
JOIN artifact_versions AS version USING (version_id)
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE revision.scope_kind = $1 AND revision.scope_id = $2 AND revision.namespace = $3 AND revision.name = $4`,
		binding.kind, binding.id, binding.namespace, binding.name).Scan(&revisions, &bytes); err != nil {
		t.Fatal(err)
	}
	if gotRevisions, gotBytes := historyCounters(t, ctx, pool, binding); gotRevisions != revisions || gotBytes != bytes {
		t.Fatalf("%v counters = %d revisions, %d bytes; retained %d and %d", binding, gotRevisions, gotBytes, revisions, bytes)
	}
}

func historyHoldCount(t *testing.T, ctx context.Context, pool *pgxpool.Pool, binding historyBinding, revision string) int {
	t.Helper()
	var holds int
	if err := pool.QueryRow(ctx, `SELECT hold_count FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4 AND revision = $5`,
		binding.kind, binding.id, binding.namespace, binding.name, revision).Scan(&holds); err != nil {
		t.Fatal(err)
	}
	return holds
}

// purgeHistoryRun purges the RunScope of runID, which releases the exact
// revisions its lineage held.
func purgeHistoryRun(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string) {
	t.Helper()
	if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		purger, err := artifactpostgres.NewPurger(tx)
		if err != nil {
			return err
		}
		return purger.PurgeRun(ctx, runID)
	}); err != nil {
		t.Fatal(err)
	}
}

// seedHistory stores count revisions of distinct content in binding, oldest
// first one millisecond apart, with accounting as writes would leave it, and
// returns their revisions in creation order.
func seedHistory(t *testing.T, ctx context.Context, pool *pgxpool.Pool, binding historyBinding, count int) []string {
	t.Helper()
	prefix := binding.namespace + "/" + binding.name + "/"
	key := []any{binding.kind, binding.id, binding.namespace, binding.name}
	content := []any{prefix, count}
	err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, statement := range []struct {
			sql       string
			arguments []any
		}{
			{`INSERT INTO artifact_scopes (scope_kind, scope_id) VALUES ($1, $2) ON CONFLICT DO NOTHING`, key[:2]},
			{`INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend)
SELECT sha256(convert_to($1 || i, 'UTF8')), convert_to($1 || i, 'UTF8'), octet_length(convert_to($1 || i, 'UTF8')), 'postgresql'
FROM generate_series(1, $2::int) AS i`, content},
			{`INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
SELECT 'version-' || $1 || i, sha256(convert_to($1 || i, 'UTF8')), 'text/plain'
FROM generate_series(1, $2::int) AS i`, content},
			{`INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id, created_at)
SELECT $1, $2, $3, $4, 'seed-' || lpad(i::text, 5, '0'), 'version-' || $5 || i,
       clock_timestamp() - interval '1 hour' + i * interval '1 millisecond'
FROM generate_series(1, $6::int) AS i`, append(key[:4:4], content...)},
			{`INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision, retained_revisions, retained_bytes)
SELECT $1, $2, $3, $4, 'seed-' || lpad(($6::int)::text, 5, '0'), $6::int,
       (SELECT sum(octet_length(convert_to($5 || i, 'UTF8'))) FROM generate_series(1, $6::int) AS i)`,
				append(key[:4:4], content...)},
		} {
			if _, err := tx.Exec(ctx, statement.sql, statement.arguments...); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	revisions := make([]string, count)
	for index := range revisions {
		revisions[index] = fmt.Sprintf("seed-%05d", index+1)
	}
	return revisions
}

// seedLargeHistory stores count revisions of size bytes each as filesystem
// content without objects.
func seedLargeHistory(t *testing.T, ctx context.Context, pool *pgxpool.Pool, binding historyBinding, count int, size int64) []string {
	t.Helper()
	key := []any{binding.kind, binding.id, binding.namespace, binding.name}
	content := []any{binding.namespace + "/" + binding.name + "/", count, size}
	err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, statement := range []struct {
			sql       string
			arguments []any
		}{
			{`INSERT INTO artifact_scopes (scope_kind, scope_id) VALUES ($1, $2) ON CONFLICT DO NOTHING`, key[:2]},
			{`INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend, object_key)
SELECT sha256(convert_to($1 || i, 'UTF8')), NULL, $3::bigint, 'filesystem',
       substr(encode(sha256(convert_to($1 || i, 'UTF8')), 'hex'), 1, 2) || '/' ||
       substr(encode(sha256(convert_to($1 || i, 'UTF8')), 'hex'), 1, 32)
FROM generate_series(1, $2::int) AS i`, content},
			{`INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
SELECT 'large-' || i, sha256(convert_to($1 || i, 'UTF8')), 'text/plain'
FROM generate_series(1, $2::int) AS i`, content[:2]},
			{`INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id, created_at)
SELECT $1, $2, $3, $4, 'large-' || i, 'large-' || i, clock_timestamp() - interval '1 hour' + i * interval '1 millisecond'
FROM generate_series(1, $5::int) AS i`, append(key[:4:4], count)},
			{`INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision, retained_revisions, retained_bytes)
VALUES ($1, $2, $3, $4, 'large-' || ($5::int), $5::int, $5::int * $6::bigint)`, append(key[:4:4], count, size)},
		} {
			if _, err := tx.Exec(ctx, statement.sql, statement.arguments...); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	revisions := make([]string, count)
	for index := range revisions {
		revisions[index] = fmt.Sprintf("large-%d", index+1)
	}
	return revisions
}

// seedUnrelatedBindings stores count single-revision bindings in the
// UserScope of user-1 that no admission may examine.
func seedUnrelatedBindings(t *testing.T, ctx context.Context, pool *pgxpool.Pool, count int) {
	t.Helper()
	err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `INSERT INTO artifact_scopes (scope_kind, scope_id) VALUES ('user', 'user-1')
ON CONFLICT DO NOTHING`); err != nil {
			return err
		}
		for _, statement := range []string{
			`INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend)
SELECT sha256(convert_to('unrelated-' || i, 'UTF8')), convert_to('unrelated-' || i, 'UTF8'),
       octet_length(convert_to('unrelated-' || i, 'UTF8')), 'postgresql'
FROM generate_series(1, $1::int) AS i`,
			`INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
SELECT 'unrelated-' || i, sha256(convert_to('unrelated-' || i, 'UTF8')), 'text/plain'
FROM generate_series(1, $1::int) AS i`,
			`INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id, created_at)
SELECT 'user', 'user-1', 'unrelated', 'binding-' || i, 'unrelated-' || i, 'unrelated-' || i,
       clock_timestamp() - interval '2 hours'
FROM generate_series(1, $1::int) AS i`,
			`INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision, retained_revisions, retained_bytes)
SELECT 'user', 'user-1', 'unrelated', 'binding-' || i, 'unrelated-' || i, 1,
       octet_length(convert_to('unrelated-' || i, 'UTF8'))
FROM generate_series(1, $1::int) AS i`,
		} {
			if _, err := tx.Exec(ctx, statement, count); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// racingBlobStore runs before once, ahead of the first content read, to put
// a concurrent writer between a reader's metadata and its content.
type racingBlobStore struct {
	blob.Storage
	before func()
}

func (s *racingBlobStore) Read(ctx context.Context, object blob.Object) ([]byte, error) {
	if before := s.before; before != nil {
		s.before = nil
		before()
	}
	return s.Storage.Read(ctx, object)
}

// A reader whose revision retention removes between its metadata and its
// filesystem content reports an exact revision as not found, never as missing
// content, and reads the new current of a versionless reference.
func TestPostgresHistoryReadRacingRetention(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	files, err := filesystem.Open(ctx, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = files.Close() })
	ctx = artifactpostgres.WithHistoryPolicy(blob.WithRuntime(ctx, blob.NewRuntime(files, nil)), artifactpostgres.NewTestHistoryPolicy(2, 1<<30))
	repository := artifactpostgres.NewRepository(pool)
	user, _ := NewService(repository).User("user-1")
	binding := historyBinding{"user", "user-1", "notes", "raced"}
	var current *string
	write := func(data string) {
		t.Helper()
		result, err := user.Write(ctx, binding.ref(), Payload{MediaType: "text/plain", Data: []byte(data)}, current)
		if err != nil {
			t.Fatal(err)
		}
		current = result.Ref.Revision
	}
	write("first revision")
	first := *current
	write("second revision")
	second := *current
	raced := func(before func()) context.Context {
		return blob.WithRuntime(ctx, blob.NewRuntime(&racingBlobStore{Storage: files, before: before}, nil))
	}

	exact := binding.ref()
	exact.Revision = &first
	if _, err := repository.Read(raced(func() { write("third revision") }), userScope(t, "user-1"), exact); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("exact read raced by retention = %v, want not found", err)
	}
	batch := []artifactpostgres.ExactReadRequest{{Scope: userScope(t, "user-1"), Ref: binding.ref()}}
	batch[0].Ref.Revision = &second
	if _, err := repository.ReadExactBatch(raced(func() { write("fourth revision") }), batch); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("exact batch read raced by retention = %v, want not found", err)
	}
	// The current revision is removed in the same commit as the write that
	// replaces it, so a versionless reader rereads the new current.
	result, err := repository.Read(raced(func() { write("fifth revision"); write("sixth revision") }), userScope(t, "user-1"), binding.ref())
	if err != nil || string(result.Payload.Data) != "sixth revision" {
		t.Fatalf("current read raced by retention = %q, %v", result.Payload.Data, err)
	}
}
