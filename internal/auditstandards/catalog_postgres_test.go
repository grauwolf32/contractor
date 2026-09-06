package auditstandards

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
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
		RequestDigest: digest([]byte("catalog-project")),
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
	if err != nil || digest(read.Payload.Data) != pinned[0].Catalog.Digest {
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
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_audit_standard_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupContext, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		if _, err := admin.Exec(cleanupContext, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop isolated Audit standard schema: %v", err)
		}
		admin.Close()
	})
	return pool
}
