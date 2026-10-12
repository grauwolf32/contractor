package scheduler

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// recordingForkRepository records the forks that reach storage; any other
// Repository method panics on the nil embedded interface.
type recordingForkRepository struct {
	artifacts.Repository
	forks []artifacts.Fork
}

func (r *recordingForkRepository) Fork(_ context.Context, fork artifacts.Fork) (artifacts.ForkResult, error) {
	r.forks = append(r.forks, fork)
	return artifacts.ForkResult{}, nil
}

// The Scheduler binds and publishes only exact Run output revisions
// (ART-R90, ART-R99): a versionless source or a publication of another slot
// never reaches the store, and the forks that do carry the output rules.
func TestOutputForksBindAndPublishOnlyExactOutputs(t *testing.T) {
	ctx := context.Background()
	repository := &recordingForkRepository{}
	service := artifacts.NewService(repository)
	revision := "revision-1"
	if _, err := bindOutput(ctx, service, "run-1", "result",
		contracts.ArtifactRef{Namespace: "builder", Name: "result"}); !errors.Is(err, artifacts.ErrExactRevisionRequired) {
		t.Fatalf("versionless output binding = %v", err)
	}
	for _, source := range []contracts.ArtifactRef{
		{Namespace: "outputs", Name: "openapi"},
		{Namespace: "builder", Name: "openapi", Revision: &revision},
		{Namespace: "outputs", Name: "other", Revision: &revision},
	} {
		if _, err := publishRunOutput(ctx, service, "run-1", "project-1", "openapi", source); err == nil {
			t.Errorf("publication of %+v was accepted", source)
		}
	}
	if len(repository.forks) != 0 {
		t.Fatalf("invalid output forks reached the store: %+v", repository.forks)
	}

	result := contracts.ArtifactRef{Namespace: "builder", Name: "result", Revision: &revision}
	if _, err := bindOutput(ctx, service, "run-1", "result", result); err != nil {
		t.Fatal(err)
	}
	output := contracts.ArtifactRef{Namespace: "outputs", Name: "openapi", Revision: &revision}
	if _, err := publishRunOutput(ctx, service, "run-1", "project-1", "openapi", output); err != nil {
		t.Fatal(err)
	}
	run, _ := artifacts.RunScope("run-1")
	project, _ := artifacts.ProjectScope("project-1")
	if len(repository.forks) != 2 {
		t.Fatalf("output forks = %+v", repository.forks)
	}
	bound, published := repository.forks[0], repository.forks[1]
	if bound.Kind != artifacts.LineageOutputBind || bound.Source != run || bound.Target != run ||
		!bound.SourceRef.SameExact(result) || bound.TargetRef != (contracts.ArtifactRef{Namespace: "outputs", Name: "result"}) ||
		bound.Frozen || bound.SourceFrozen || bound.Replay {
		t.Fatalf("output binding fork = %+v", bound)
	}
	if published.Kind != artifacts.LineageProjectOutputPublish || published.Source != run || published.Target != project ||
		!published.SourceRef.SameExact(output) || !published.SourceFrozen ||
		published.TargetRef != (contracts.ArtifactRef{Namespace: "outputs", Name: "openapi"}) ||
		published.Frozen || published.Replay {
		t.Fatalf("Project publication fork = %+v", published)
	}
}
