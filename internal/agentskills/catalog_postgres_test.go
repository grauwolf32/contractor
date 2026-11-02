package agentskills

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestCatalogPostgresConcurrentInitializationIsCreateOnly(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSkillCatalogPool(t, ctx)
	plan := bundledPlan(t, map[string]string{"alpha": "Alpha."})
	service := artifacts.NewService(artifactpostgres.NewRepository(pool))
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

// On the real store, an exact Skill source that history retention removed is
// the permanent skill_artifact_not_found at the Run creation lock and at the
// initialization read (RUN-R27), while a cancelled request, which proves
// nothing about the revision, stays the retryable skill_artifact_unavailable.
func TestCatalogPostgresRunSourceRemovedByRetentionIsNotFound(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSkillCatalogPool(t, ctx)
	policy, err := artifactpostgres.NewHistoryPolicy(artifactpostgres.HistoryRevisionsFloor, artifactpostgres.HistoryBytesFloor)
	if err != nil {
		t.Fatal(err)
	}
	ctx = artifactpostgres.WithHistoryPolicy(ctx, policy)
	service := artifacts.NewService(artifactpostgres.NewRepository(pool))
	catalog, err := NewCatalog(service)
	if err != nil {
		t.Fatal(err)
	}
	owner, _ := service.User("owner-1")
	binding := artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "review"}
	var revision *string
	var selections [][]RunSkillSnapshot
	for _, description := range []string{"Pinned.", "Newer.", "Newest."} {
		written, err := owner.Write(ctx, binding, artifacts.Payload{
			MediaType: MediaType, Data: canonicalTestPackage(t, "review", description),
		}, revision)
		if err != nil {
			t.Fatal(err)
		}
		revision = written.Ref.Revision
		selected, err := catalog.SelectRunSources(ctx, "owner-1", []artifacts.ArtifactRef{binding})
		if err != nil || len(selected) != 1 || selected[0].Source == nil {
			t.Fatalf("selected source = (%+v, %v)", selected, err)
		}
		selections = append(selections, selected)
	}
	gone, current := selections[0], selections[2]
	if _, err := owner.Read(ctx, *gone[0].Source); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("read of the revision retention removed = %v, want not found", err)
	}
	templateSets := [][]string{{"review"}}
	assertRunSkillError(t, catalog.RequireRunSources(ctx, "owner-1", gone), CodeArtifactNotFound, "review", false)
	_, err = catalog.InitializeRun(ctx, "owner-1", "run-1", gone, templateSets)
	assertRunSkillError(t, err, CodeArtifactNotFound, "review", false)

	cancelled, stop := context.WithCancel(ctx)
	stop()
	assertRunSkillError(t, catalog.RequireRunSources(cancelled, "owner-1", current), CodeArtifactUnavailable, "review", true)
	_, err = catalog.InitializeRun(cancelled, "owner-1", "run-1", current, templateSets)
	assertRunSkillError(t, err, CodeArtifactUnavailable, "review", true)
}

func isolatedSkillCatalogPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}
