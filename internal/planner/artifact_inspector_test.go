package planner

import (
	"context"
	"errors"
	"testing"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	artifactpgtest "github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	inline "github.com/grauwolf32/contractor/internal/artifacts/blob/backends/postgres"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

// Only Metadata may be called: a payload Read would both panic and require a
// transfer slot in the saturated context used by this test.
// inspectorRunID is the only Run whose artifacts the repository holds.
const inspectorRunID = "run-1"

type inspectorMetadataRepository struct {
	artifacts.Repository
	entries map[string]artifacts.Metadata // by namespace/name/revision of Run inspectorRunID
	queries int
}

func (r *inspectorMetadataRepository) Read(context.Context, artifacts.Scope, contracts.ArtifactRef) (artifacts.ReadResult, error) {
	panic("Planner inspection read artifact payload")
}

func (r *inspectorMetadataRepository) Metadata(
	_ context.Context, scope artifacts.Scope, ref contracts.ArtifactRef,
) (artifacts.Metadata, error) {
	r.queries++
	if want, err := artifacts.RunScope(inspectorRunID); err != nil || scope != want || ref.ValidateExact() != nil {
		return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
	}
	metadata, ok := r.entries[ref.Namespace+"/"+ref.Name+"/"+*ref.Revision]
	if !ok {
		return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
	}
	return metadata, nil
}

func saturatedInspectorContext(t *testing.T) context.Context {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	t.Helper()
	ctx := t.Context()
	artifactRuntime = blob.NewRuntime(inline.Store{}, nil)
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	for range 4 {
		_, release, err := artifactStorage.AcquireTransfer(ctx)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(release)
	}
	if _, _, err := artifactStorage.AcquireTransfer(ctx); !errors.Is(err, blob.ErrTransferCapacity) {
		t.Fatalf("transfer gate was not saturated: %v", err)
	}
	return ctx
}

func TestArtifactInspectorsUseExactMetadataWithTransferSlotsSaturated(t *testing.T) {
	ctx := saturatedInspectorContext(t)
	revision := "rev-1"
	ref := contracts.ArtifactRef{Namespace: "builder", Name: "report", Revision: &revision}
	repository := &inspectorMetadataRepository{entries: map[string]artifacts.Metadata{
		"builder/report/rev-1": {Ref: ref, MediaType: "application/json"},
	}}
	service := artifacts.NewService(repository)
	inspector, err := NewArtifactServiceInspector(service)
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := inspector.Inspect(ctx, "run-1", ref)
	if err != nil || metadata.MediaType != "application/json" {
		t.Fatalf("exact metadata inspection = (%+v, %v)", metadata, err)
	}
	candidate := stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: "done",
		Artifacts: map[string]contracts.ArtifactRef{"report": ref},
	}
	contract := map[string]resolved.ArtifactSlot{
		"report": {Required: true, MediaTypes: []string{"application/json"}},
	}
	if failure := ValidateCandidate(ctx, "run-1", contract, candidate, inspector); failure != nil {
		t.Fatalf("candidate failed under saturated transfer gate: %v", failure)
	}
	if repository.queries != 2 {
		t.Fatalf("metadata query count = %d, want 2", repository.queries)
	}
	if _, err := inspector.Inspect(ctx, "run-2", ref); !artifacts.IsExactReferenceRejection(err) {
		t.Fatalf("foreign RunScope inspection = %v, want a rejected reference", err)
	}
	missing := ref
	missing.Name = "missing"
	if _, err := inspector.Inspect(ctx, "run-1", missing); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("missing artifact = %v", err)
	}
	if _, err := inspector.Inspect(ctx, "run-1", contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name}); !artifacts.IsExactReferenceRejection(err) {
		t.Fatalf("binding head inspection = %v, want a rejected reference", err)
	}
	wrong := "rev-2"
	repository.entries["builder/report/rev-1"] = artifacts.Metadata{
		Ref:       contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &wrong},
		MediaType: "application/json",
	}
	if _, err := inspector.Inspect(ctx, "run-1", ref); !errors.Is(err, blob.ErrIntegrity) {
		t.Fatalf("metadata for another revision = %v, want ErrArtifactIntegrity", err)
	}
}
