package auditstandards

import (
	"context"
	"errors"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestCatalogPostgresConcurrentSeedAndExactRetention(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedCatalogPool(t, ctx, databaseURL)
	root := t.TempDir()
	writePackageSource(t, root, "example-v1", validDocument("example", "1"))
	plan, err := DiscoverBundled(root)
	if err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	catalog, _ := NewCatalog(service)

	var wait sync.WaitGroup
	errorsFound := make(chan error, 8)
	for range 8 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, initializeErr := catalog.Initialize(ctx, "catalog-owner", plan)
			errorsFound <- initializeErr
		}()
	}
	wait.Wait()
	close(errorsFound)
	for initializeErr := range errorsFound {
		if initializeErr != nil {
			t.Fatalf("concurrent initialize: %v", initializeErr)
		}
	}
	var revisions int
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM artifact_binding_revisions
 WHERE scope_kind = 'user' AND scope_id = 'catalog-owner'
   AND namespace = 'audit-standards'`).Scan(&revisions); err != nil || revisions != 1 {
		t.Fatalf("catalog revisions = %d, error=%v", revisions, err)
	}

	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "catalog-project", OwnerID: "catalog-owner", Kind: projectstore.KindProject,
		Name: "Catalog project", IdempotencyKey: "create-catalog-project",
		RequestDigest: contentdigest.Bytes([]byte("catalog-project")),
	})
	if err != nil {
		t.Fatal(err)
	}
	pinned, err := catalog.Pin(ctx, project.OwnerID, project.ProjectID, "audit-catalog-test",
		[]Reference{{Scheme: "example", Version: "1"}})
	if err != nil || len(pinned) != 1 || pinned[0].Retained.Artifact.Revision == nil {
		t.Fatalf("pin = (%+v, %v)", pinned, err)
	}
	retained, err := service.Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	read, err := retained.Read(ctx, pinned[0].Retained.Artifact)
	if err != nil || contentdigest.Bytes(read.Payload.Data) != pinned[0].Catalog.Digest {
		t.Fatalf("retained package integrity = (%+v, %v)", read, err)
	}
	var frozen bool
	if err := pool.QueryRow(ctx, `
SELECT frozen FROM artifact_bindings
 WHERE scope_kind = 'project' AND scope_id = $1
   AND namespace = 'audit-catalog-test' AND name = $2`,
		project.ProjectID, pinned[0].Retained.Artifact.Name).Scan(&frozen); err != nil || !frozen {
		t.Fatalf("retained package frozen = %t, error=%v", frozen, err)
	}
}

func isolatedCatalogPool(
	t *testing.T, ctx context.Context, databaseURL string,
) *pgxpool.Pool {
	t.Helper()
	pool := postgrestest.NewPool(t, ctx, databaseURL, postgrestest.Options{})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}

func TestCatalogPostgresListRejectsAnOverfullCatalogWithABoundedRead(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedCatalogPool(t, ctx, databaseURL)
	root := t.TempDir()
	writePackageSource(t, root, "example-v1", validDocument("example", "1"))
	plan, err := DiscoverBundled(root)
	if err != nil {
		t.Fatal(err)
	}
	catalog, _ := NewCatalog(artifacts.NewService(artifacts.NewPostgresRepository(pool)))
	if _, err := catalog.Initialize(ctx, "catalog-owner", plan); err != nil {
		t.Fatal(err)
	}
	if listed, err := catalog.List(ctx, "catalog-owner"); err != nil || len(listed) != 1 {
		t.Fatalf("catalog list = (%d, %v)", len(listed), err)
	}
	// Bindings beyond the package limit fail the listing before any of them
	// is read or validated.
	_, err = pool.Exec(ctx, `WITH bindings AS (
 INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision, retained_revisions, retained_bytes)
 SELECT 'user', 'catalog-owner', 'audit-standards', 'zz-extra-' || lpad(i::text, 4, '0'), 'bulk-rev', 1, 0
 FROM generate_series(1, $1::int) AS i
 RETURNING scope_kind, scope_id, namespace, name, current_revision
) INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
 SELECT scope_kind, scope_id, namespace, name, current_revision, (SELECT version_id FROM artifact_versions LIMIT 1)
 FROM bindings`, MaximumPackages+500)
	if err != nil {
		t.Fatal(err)
	}
	_, err = catalog.List(ctx, "catalog-owner")
	var validation *ValidationError
	if !errors.As(err, &validation) || validation.Code != CodeLimitExceeded {
		t.Fatalf("overfull catalog list error = %v", err)
	}
}
