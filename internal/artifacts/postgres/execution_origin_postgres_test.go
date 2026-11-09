package postgres_test

import (
	"context"
	"reflect"
	"testing"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
)

func TestPostgresExecutionOriginSurvivesForkAndDistinguishesIndependentWrites(t *testing.T) {
	ctx := context.Background()
	pool := isolatedArtifactPool(t, ctx)
	createArtifactRun(t, ctx, pool, "origin-run", false)
	repo := artifactpgtest.NewStorage().Repository(pool)
	service := NewService(repo)
	run, _ := service.Run("origin-run")
	user, _ := service.User("user-1")
	origin := &ExecutionOrigin{RunID: "origin-run", StageExecutionID: "origin-stage", AllocationID: "origin-allocation", LogicalAgentName: "scanner"}
	first, err := run.Write(ctx, ArtifactRef{Namespace: "scanner", Name: "first"}, Payload{MediaType: "application/json", Data: []byte(`{}`), ExecutionOrigin: origin}, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := run.Write(ctx, ArtifactRef{Namespace: "scanner", Name: "second"}, Payload{MediaType: "application/json", Data: []byte(`{}`), ExecutionOrigin: origin}, nil)
	if err != nil {
		t.Fatal(err)
	}
	uploaded, err := user.Write(ctx, ArtifactRef{Namespace: "uploaded", Name: "copy"}, Payload{MediaType: "application/json", Data: []byte(`{}`)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	scope, _ := RunScope("origin-run")
	expected := *origin
	expected.Artifact = first.Ref
	got, firstID, err := repo.ExecutionOrigin(ctx, scope, first.Ref)
	if err != nil || !reflect.DeepEqual(got, &expected) || firstID == "" {
		t.Fatalf("origin: %+v %q %v", got, firstID, err)
	}
	_, secondID, err := repo.ExecutionOrigin(ctx, scope, second.Ref)
	if err != nil || firstID == secondID {
		t.Fatalf("independent writes share identity: %v", err)
	}
	copied, err := service.Fork(ctx, Fork{Source: scope, SourceRef: first.Ref, Target: scope, TargetRef: ArtifactRef{Namespace: "copy", Name: "first"}, Kind: LineageOutputBind, Frozen: true})
	if err != nil {
		t.Fatal(err)
	}
	got, copyID, err := repo.ExecutionOrigin(ctx, scope, copied.TargetRef)
	if err != nil || copyID != firstID || !reflect.DeepEqual(got, &expected) {
		t.Fatalf("fork: %+v %q %v", got, copyID, err)
	}
	userScope, _ := UserScope("user-1")
	got, _, err = repo.ExecutionOrigin(ctx, userScope, uploaded.Ref)
	if err != nil || got != nil {
		t.Fatalf("upload impersonates execution: %+v %v", got, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE artifact_versions SET execution_origin='{}'::jsonb WHERE version_id=$1`, firstID); err == nil {
		t.Fatal("execution provenance was mutable")
	}
}
