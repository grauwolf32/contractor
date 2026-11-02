package runstore

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// storeTestRunArtifact writes a revision of namespace/name in the RunScope of
// runID through db, as a Worker does, and returns its exact ref. A
// StageExecution holds every exact ref it records, so each must name a
// stored revision (DEL-R62).
func storeTestRunArtifact(
	t *testing.T, ctx context.Context, db persistencepostgres.DBTX, runID, namespace, name string,
) contracts.ArtifactRef {
	t.Helper()
	ref, err := artifactstest.WriteRunArtifact(ctx, artifacts.NewService(artifactpgtest.NewStorage().Repository(db)), runID,
		contracts.ArtifactRef{Namespace: namespace, Name: name},
		artifacts.Payload{MediaType: "text/plain", Data: []byte(namespace + "/" + name)})
	if err != nil {
		t.Fatal(err)
	}
	return ref
}

// storeTestRunInput forks a UserScope source of the test owner into
// inputs/<slot> of runID, as Run creation does, and returns its exact ref.
func storeTestRunInput(t *testing.T, ctx context.Context, db persistencepostgres.DBTX, runID, slot string) contracts.ArtifactRef {
	t.Helper()
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(db))
	user, err := service.User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	source, err := user.Write(ctx, contracts.ArtifactRef{Namespace: "sources", Name: runID + "-" + slot},
		artifacts.Payload{MediaType: "text/plain", Data: []byte(slot)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	forked, err := artifactstest.ForkInput(ctx, service, "user-1", source.Ref, runID, slot)
	if err != nil {
		t.Fatal(err)
	}
	return forked.TargetRef
}
