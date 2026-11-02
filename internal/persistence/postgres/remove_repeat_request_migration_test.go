package postgres_test

import (
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/persistence/migrations"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

func TestPostgresRemoveRepeatRequestMigrationPreservesRunAndSharedArtifacts(t *testing.T) {
	ctx := t.Context()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{})
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{
		RunID: "repeat-migration", OwnerID: "owner", WorkflowName: "review", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: []byte(`{}`),
		LaneClass: "model", RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	scope, _ := artifacts.RunScope("repeat-migration")
	for _, entry := range []struct{ namespace, name, body string }{
		{"inputs", "source", "shared bytes"},
		{"contractor-system", "repeat-request", "shared bytes"},
		{"contractor-system", "orphan", "obsolete unique bytes"},
	} {
		if _, err := service.Create(ctx, scope, contracts.ArtifactRef{Namespace: entry.namespace, Name: entry.name}, artifacts.Payload{MediaType: "text/plain", Data: []byte(entry.body)}); err != nil {
			t.Fatal(err)
		}
	}
	contents, err := migrations.Files.ReadFile("000003_remove_repeat_request.sql")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, string(contents)); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.GetRun(ctx, "repeat-migration"); err != nil {
		t.Fatal(err)
	}
	view, _ := service.Run("repeat-migration")
	read, err := view.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil || string(read.Payload.Data) != "shared bytes" {
		t.Fatalf("ordinary input lost: %+v %v", read, err)
	}
	var bindings, revisions, versions, blobs int
	if err := pool.QueryRow(ctx, `SELECT
 (SELECT count(*) FROM artifact_bindings), (SELECT count(*) FROM artifact_binding_revisions),
 (SELECT count(*) FROM artifact_versions), (SELECT count(*) FROM artifact_blobs)`).Scan(&bindings, &revisions, &versions, &blobs); err != nil {
		t.Fatal(err)
	}
	if bindings != 1 || revisions != 1 || versions != 1 || blobs != 1 {
		t.Fatalf("obsolete storage retained: bindings=%d revisions=%d versions=%d blobs=%d", bindings, revisions, versions, blobs)
	}
	var definition string
	if err := pool.QueryRow(ctx, `SELECT pg_get_indexdef('artifact_bindings_user_listing_idx'::regclass)`).Scan(&definition); err != nil || strings.Contains(definition, "contractor-system") {
		t.Fatalf("obsolete listing policy retained: %s %v", definition, err)
	}
}
