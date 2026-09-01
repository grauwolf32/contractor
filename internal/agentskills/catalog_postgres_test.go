package agentskills

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
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestCatalogPostgresConcurrentInitializationIsCreateOnly(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSkillCatalogPool(t, ctx)
	plan := bundledPlan(t, map[string]string{"alpha": "Alpha."})
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	catalog, err := NewCatalog(service)
	if err != nil {
		t.Fatal(err)
	}

	var wait sync.WaitGroup
	errorsFound := make(chan error, 8)
	for range 8 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, initializeErr := catalog.Initialize(ctx, "postgres-owner", plan)
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
SELECT count(*)
FROM artifact_binding_revisions
WHERE scope_kind = 'user' AND scope_id = 'postgres-owner'
  AND namespace = 'skills' AND name = 'alpha'`).Scan(&revisions); err != nil {
		t.Fatal(err)
	}
	if revisions != 1 {
		t.Fatalf("concurrent initialization created %d revisions, want 1", revisions)
	}
	outcomes, err := catalog.Initialize(ctx, "postgres-owner", plan)
	if err != nil || len(outcomes) != 1 || outcomes[0].Status != SeedInSync {
		t.Fatalf("restart outcomes = (%+v, %v)", outcomes, err)
	}

	owner, _ := service.User("postgres-owner")
	current, err := owner.Read(ctx, artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "alpha"})
	if err != nil {
		t.Fatal(err)
	}
	updated, err := owner.Write(
		ctx,
		artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "alpha"},
		artifacts.Payload{MediaType: "application/zip", Data: []byte("operator update")},
		current.Ref.Revision,
	)
	if err != nil {
		t.Fatal(err)
	}
	outcomes, err = catalog.Initialize(ctx, "postgres-owner", plan)
	if err != nil || outcomes[0].Status != SeedDrift {
		t.Fatalf("drift restart = (%+v, %v)", outcomes, err)
	}
	after, err := owner.Read(ctx, artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "alpha"})
	if err != nil || after.Ref.Revision == nil || updated.Ref.Revision == nil || *after.Ref.Revision != *updated.Ref.Revision {
		t.Fatalf("drift revision changed: (%+v, %v)", after, err)
	}
}

func isolatedSkillCatalogPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
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
	schema := "contractor_skill_catalog_" + hex.EncodeToString(random)
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
			t.Logf("drop isolated SkillCatalog schema: %v", err)
		}
		admin.Close()
	})
	return pool
}
