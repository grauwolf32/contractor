package artifacts_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// A ProjectScope purge commits at most one purge batch per transaction. Each
// committed batch leaves every remaining binding with its current revision,
// even where a binding's revisions straddle two batches, collects the
// versions and blobs the batch leaves unreferenced, including their
// filesystem objects, and keeps content another scope still references. The
// call after the last batch removes the scope.
func TestPostgresIntegrationProjectScopePurgeCommitsBoundedBatches(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx, objects := purgeBlobContext(t, ctx, pool)
	const batch = 4
	SetPurgeBatchSize(t, batch)
	const projectID = "purge-batches"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Purge batches", IdempotencyKey: "create-purge-batches",
		RequestDigest: "sha256:" + strings.Repeat("c", 64),
	}); err != nil {
		t.Fatal(err)
	}
	service := NewService(NewPostgresRepository(pool))
	user, _ := service.User("user-1")
	project, _ := service.Project(projectID)
	shared := Payload{MediaType: "text/plain", Data: []byte("shared bytes")}
	kept, err := user.Write(ctx, ArtifactRef{Namespace: "keep", Name: "shared"}, shared, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Five bindings of three revisions each: batches of four split them.
	for binding := range 5 {
		var revision *string
		for version := range 3 {
			payload := Payload{MediaType: "text/plain", Data: fmt.Appendf(nil, "binding %d version %d", binding, version)}
			if binding == 2 && version == 1 {
				payload = shared
			}
			written, err := project.Write(ctx, ArtifactRef{Namespace: "notes", Name: fmt.Sprintf("note-%d", binding)}, payload, revision)
			if err != nil {
				t.Fatal(err)
			}
			revision = written.Ref.Revision
		}
	}
	projectObjects := blobObjectKeys(t, ctx, pool, `
SELECT DISTINCT blob.object_key FROM artifact_binding_revisions AS revision
JOIN artifact_versions AS version USING (version_id)
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE revision.scope_kind = 'project' AND blob.object_key IS NOT NULL
  AND blob.payload IS DISTINCT FROM 'shared bytes'::bytea
  AND NOT EXISTS (
      SELECT 1 FROM artifact_binding_revisions AS other
      JOIN artifact_versions AS other_version USING (version_id)
      WHERE other.scope_kind = 'user' AND other_version.blob_sha256 = blob.sha256)`)
	sharedObjects := blobObjectKeys(t, ctx, pool, `
SELECT blob.object_key FROM artifact_versions AS version
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
JOIN artifact_binding_revisions AS revision USING (version_id)
WHERE revision.scope_kind = 'user' AND blob.object_key IS NOT NULL`)

	revisionsLeft := func() int {
		t.Helper()
		var count int
		if err := pool.QueryRow(ctx, `
SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind = 'project' AND scope_id = $1`,
			projectID).Scan(&count); err != nil {
			t.Fatal(err)
		}
		return count
	}
	total := revisionsLeft()
	if total != 15 {
		t.Fatalf("seeded %d Project revisions, want 15", total)
	}
	steps := 0
	for gone := false; !gone; steps++ {
		if steps > total {
			t.Fatalf("ProjectScope purge did not finish after %d transactions", steps)
		}
		before := revisionsLeft()
		if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			purger, err := NewPostgresPurger(tx)
			if err != nil {
				return err
			}
			gone, err = purger.PurgeProject(ctx, projectID)
			return err
		}); err != nil {
			t.Fatalf("ProjectScope purge transaction %d: %v", steps, err)
		}
		if removed := before - revisionsLeft(); removed > batch || (!gone && removed == 0) {
			t.Fatalf("ProjectScope purge transaction %d removed %d revisions, want 1 to %d", steps, removed, batch)
		}
	}
	if want := (total+batch-1)/batch + 1; steps != want {
		t.Fatalf("ProjectScope purge took %d transactions, want %d", steps, want)
	}
	var left int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM artifact_scopes WHERE scope_kind = 'project')
     + (SELECT count(*) FROM artifact_bindings WHERE scope_kind = 'project')
     + (SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind = 'project')
     + (SELECT count(*) FROM artifact_versions) + (SELECT count(*) FROM artifact_blobs)`,
	).Scan(&left); err != nil {
		t.Fatal(err)
	}
	if left != 2 {
		t.Fatalf("ProjectScope purge left %d rows, want only the shared version and blob", left)
	}
	if read, err := user.Read(ctx, kept.Ref); err != nil || string(read.Payload.Data) != "shared bytes" {
		t.Fatalf("shared UserScope content after the ProjectScope purge = (%q, %v)", read.Payload.Data, err)
	}
	for _, key := range projectObjects {
		if _, err := os.Stat(filepath.Join(objects, key)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("filesystem object %s of a purged blob: %v", key, err)
		}
	}
	for _, key := range sharedObjects {
		if _, err := os.Stat(filepath.Join(objects, key)); err != nil {
			t.Fatalf("filesystem object %s of the shared blob: %v", key, err)
		}
	}
	if objects != "" && (len(projectObjects) != 14 || len(sharedObjects) != 1) {
		t.Fatalf("filesystem objects = %d Project and %d shared, want 14 and 1", len(projectObjects), len(sharedObjects))
	}
}

// purgeBlobContext installs the filesystem backend when the test run selects
// it and returns the directory of its objects, or "" for PostgreSQL payloads.
func purgeBlobContext(t *testing.T, ctx context.Context, pool *pgxpool.Pool) (context.Context, string) {
	t.Helper()
	if os.Getenv("CONTRACTOR_TEST_ARTIFACT_BACKEND") != "filesystem" {
		return ctx, ""
	}
	if err := ClaimBlobBackend(ctx, pool, BlobFilesystem); err != nil {
		t.Fatal(err)
	}
	path := t.TempDir()
	files, err := OpenFilesystemBlobStore(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = files.Close() })
	return WithBlobRuntime(ctx, NewBlobRuntime(files, nil)), path
}

func blobObjectKeys(t *testing.T, ctx context.Context, pool *pgxpool.Pool, sql string) []string {
	t.Helper()
	rows, err := pool.Query(ctx, sql)
	if err != nil {
		t.Fatal(err)
	}
	keys, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatal(err)
	}
	return keys
}
