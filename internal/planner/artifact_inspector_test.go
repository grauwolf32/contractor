package planner

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// Only Metadata may be called: a payload Read would both panic and require a
// transfer slot in the saturated context used by this test.
type inspectorMetadataRepository struct {
	artifacts.Repository
	artifacts.QueryRepository
	entries map[string]artifacts.Metadata
	queries int
}

func (r *inspectorMetadataRepository) Read(context.Context, artifacts.Scope, contracts.ArtifactRef) (artifacts.ReadResult, error) {
	panic("Planner inspection read artifact payload")
}

func (r *inspectorMetadataRepository) Metadata(
	_ context.Context, scope artifacts.Scope, ref contracts.ArtifactRef,
) (artifacts.Metadata, error) {
	r.queries++
	if scope.Kind() != artifacts.ScopeRun || ref.ValidateExact() != nil {
		return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
	}
	metadata, ok := r.entries[scope.ID()+"/"+ref.Namespace+"/"+ref.Name+"/"+*ref.Revision]
	if !ok {
		return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
	}
	return metadata, nil
}

func saturatedInspectorContext(t *testing.T) context.Context {
	t.Helper()
	ctx := artifacts.WithBlobRuntime(t.Context(), artifacts.NewBlobRuntime(artifacts.PostgresBlobStore{}, nil))
	for range 4 {
		_, release, err := artifacts.AcquireTransfer(ctx)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(release)
	}
	if _, _, err := artifacts.AcquireTransfer(ctx); !errors.Is(err, artifacts.ErrTransferCapacity) {
		t.Fatalf("transfer gate was not saturated: %v", err)
	}
	return ctx
}

func TestArtifactInspectorsUseExactMetadataWithTransferSlotsSaturated(t *testing.T) {
	ctx := saturatedInspectorContext(t)
	revision := "rev-1"
	ref := contracts.ArtifactRef{Namespace: "builder", Name: "report", Revision: &revision}
	repository := &inspectorMetadataRepository{entries: map[string]artifacts.Metadata{
		"run-1/builder/report/rev-1": {Ref: ref, MediaType: "application/json"},
	}}
	service := artifacts.NewService(repository)
	serviceInspector, err := NewArtifactServiceInspector(service)
	if err != nil {
		t.Fatal(err)
	}
	runStore, err := service.Run("run-1")
	if err != nil {
		t.Fatal(err)
	}
	runInspector, err := NewRunArtifactInspector("run-1", runStore)
	if err != nil {
		t.Fatal(err)
	}
	for _, inspector := range []ArtifactInspector{serviceInspector, runInspector} {
		metadata, err := inspector.Inspect(ctx, "run-1", ref)
		if err != nil || metadata.MediaType != "application/json" {
			t.Fatalf("exact metadata inspection = (%+v, %v)", metadata, err)
		}
		candidate := contracts.StageContentResult{
			APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: "done",
			Artifacts: map[string]contracts.ArtifactRef{"report": ref},
		}
		contract := map[string]workflowconfig.ArtifactSlot{
			"report": {Required: true, MediaTypes: []string{"application/json"}},
		}
		if failure := ValidateCandidate(ctx, "run-1", contract, candidate, inspector); failure != nil {
			t.Fatalf("candidate failed under saturated transfer gate: %v", failure)
		}
	}
	if repository.queries != 4 {
		t.Fatalf("metadata query count = %d, want 4", repository.queries)
	}
	for _, inspector := range []ArtifactInspector{serviceInspector, runInspector} {
		if _, err := inspector.Inspect(ctx, "run-2", ref); err == nil {
			t.Fatal("inspector accepted foreign RunScope")
		}
		missing := ref
		missing.Name = "missing"
		if _, err := inspector.Inspect(ctx, "run-1", missing); !errors.Is(err, artifacts.ErrArtifactNotFound) {
			t.Fatalf("missing artifact = %v", err)
		}
		if _, err := inspector.Inspect(ctx, "run-1", contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name}); err == nil {
			t.Fatal("inspector accepted a binding head without exact revision")
		}
	}
	wrong := "rev-2"
	repository.entries["run-1/builder/report/rev-1"] = artifacts.Metadata{
		Ref:       contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &wrong},
		MediaType: "application/json",
	}
	for _, inspector := range []ArtifactInspector{serviceInspector, runInspector} {
		if _, err := inspector.Inspect(ctx, "run-1", ref); err == nil {
			t.Fatal("inspector accepted metadata for another revision")
		}
	}
}
