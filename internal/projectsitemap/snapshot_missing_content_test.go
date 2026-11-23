package projectsitemap

import (
	"crypto/sha256"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPostgresSitemapSnapshotMissingContentReportsStorageFailure(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	pool := sitemapPool(t)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "consumer")
	path := t.TempDir()
	files, err := filesystem.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	artifactRuntime = blob.NewRuntime(files, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	stores := artifacts.NewService(artifactStorage.Repository(pool))
	project, _ := artifacts.ProjectScope("project")
	payload := artifacts.Payload{MediaType: "application/json", Data: []byte(`{"openapi":"3.1.0"}`)}
	source, err := stores.Create(ctx, project, contracts.ArtifactRef{Namespace: "schemas", Name: "api"}, payload)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactstest.ForkProjectInput(ctx, stores, "project", source.Ref, "consumer", "api"); err != nil {
		t.Fatal(err)
	}
	service := NewService(pool, artifactStorage)
	exact := contracts.ExactContent(source.Ref, payload.MediaType, payload.Data)
	_, err = service.Mutate(ctx, Authority{OwnerID: "owner", ProjectID: "project", SourceID: "schema", Kind: "openapi"}, "import", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: "schema", Artifact: &exact, Rules: []Rule{{SourceID: "schema", Origin: "https://example.test", Method: "GET", Path: "/declared"}}}})
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(payload.Data)
	var objectKey string
	if err := pool.QueryRow(ctx, `SELECT object_key FROM artifact_blobs WHERE sha256=$1`, digest[:]).Scan(&objectKey); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(path, objectKey)); err != nil {
		t.Fatal(err)
	}
	policy := readerPolicy()
	policy.SnapshotInputs = []string{"api"}
	access, err := service.AllocationAccess(ctx, "consumer", "research", policy)
	if err != nil {
		t.Fatal(err)
	}
	view, err := readSitemap(service, ctx, access.ProjectID, "")
	if err != nil || len(view.Document.Groups) != 1 || view.Document.Groups[0].Route != "/declared" {
		t.Fatalf("missing source bytes hid map knowledge: %v", err)
	}
	selection := SnapshotSelection{Revision: view.Revision, Groups: []string{view.Document.Groups[0].ID}, RequireReferencedInputs: true}
	if _, err := service.SnapshotAllocation(ctx, access, "required", selection); !errors.Is(err, blob.ErrMissing) {
		t.Fatalf("required missing source: %v", err)
	}
	selection.RequireReferencedInputs = false
	if _, err := service.SnapshotAllocation(ctx, access, "optional", selection); !errors.Is(err, blob.ErrMissing) {
		t.Fatalf("optional selection concealed storage loss: %v", err)
	}
	var artifactsCreated int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind='run' AND scope_id='consumer' AND namespace='research'`).Scan(&artifactsCreated); err != nil || artifactsCreated != 0 {
		t.Fatalf("failed snapshots left partial input forks: %d %v", artifactsCreated, err)
	}
}
