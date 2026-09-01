package agentskills

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestCatalogRunInitializationKeepsSelectedExactSourceAfterCurrentUpdate(t *testing.T) {
	ctx := context.Background()
	repository := newMemoryArtifactRepository()
	service := artifacts.NewService(repository)
	catalog, err := NewCatalog(service)
	if err != nil {
		t.Fatal(err)
	}
	owner, _ := service.User("owner-1")
	packageA := canonicalTestPackage(t, "review", "Package A.")
	created, err := owner.Write(
		ctx,
		artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "review"},
		artifacts.Payload{MediaType: MediaType, Data: packageA},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	selected, err := catalog.SelectRunSources(ctx, "owner-1", []contracts.ArtifactRef{{
		Namespace: SkillNamespace, Name: "review",
	}})
	if err != nil || len(selected) != 1 || selected[0].Source == nil ||
		*selected[0].Source.Revision != *created.Ref.Revision {
		t.Fatalf("selected source = (%+v, %v)", selected, err)
	}
	packageB := canonicalTestPackage(t, "review", "Package B.")
	updated, err := owner.Write(
		ctx,
		artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "review"},
		artifacts.Payload{MediaType: MediaType, Data: packageB},
		created.Ref.Revision,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := catalog.PinRunSources(ctx, "owner-1", "run-1", selected); err != nil {
		t.Fatal(err)
	}
	initialized, err := catalog.InitializeRun(
		ctx, "owner-1", "run-1", selected, [][]string{{"review"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if initialized[0].Artifact == nil || initialized[0].PackageDigest != selected[0].SourceDigest ||
		repository.forkCalls != 1 {
		t.Fatalf("initialized snapshot = %+v, fork calls = %d", initialized, repository.forkCalls)
	}
	run, _ := service.Run("run-1")
	forked, err := run.Read(ctx, *initialized[0].Artifact)
	if err != nil || !bytes.Equal(forked.Payload.Data, packageA) || bytes.Equal(forked.Payload.Data, packageB) {
		t.Fatalf("forked selected bytes = (%d bytes, %v)", len(forked.Payload.Data), err)
	}
	current, err := owner.Read(ctx, artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "review"})
	if err != nil || current.Ref.Revision == nil || *current.Ref.Revision != *updated.Ref.Revision {
		t.Fatalf("owner current = (%+v, %v)", current.Ref, err)
	}

	again, err := catalog.InitializeRun(ctx, "owner-1", "run-1", selected, [][]string{{"review"}})
	if err != nil || repository.forkCalls != 1 || *again[0].Artifact.Revision != *initialized[0].Artifact.Revision {
		t.Fatalf("idempotent initialization = (%+v, %v), forks=%d", again, err, repository.forkCalls)
	}
}

func TestCatalogRunSelectionPersistsMissingAndFailsDeterministically(t *testing.T) {
	repository := newMemoryArtifactRepository()
	catalog, _ := NewCatalog(artifacts.NewService(repository))
	selected, err := catalog.SelectRunSources(context.Background(), "owner-1", []contracts.ArtifactRef{{
		Namespace: SkillNamespace, Name: "missing",
	}})
	if err != nil || len(selected) != 1 || selected[0].Source != nil {
		t.Fatalf("missing selection = (%+v, %v)", selected, err)
	}
	_, err = catalog.InitializeRun(
		context.Background(), "owner-1", "run-missing", selected, [][]string{{"missing"}},
	)
	assertRunSkillError(t, err, CodeArtifactNotFound, "missing", false)
}

func TestCatalogRunInitializationRejectsWrongMediaType(t *testing.T) {
	ctx := context.Background()
	repository := newMemoryArtifactRepository()
	service := artifacts.NewService(repository)
	catalog, _ := NewCatalog(service)
	owner, _ := service.User("owner-1")
	payload := canonicalTestPackage(t, "review", "Review.")
	if _, err := owner.Write(
		ctx,
		artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "review"},
		artifacts.Payload{MediaType: "application/zip", Data: payload},
		nil,
	); err != nil {
		t.Fatal(err)
	}
	selected, err := catalog.SelectRunSources(ctx, "owner-1", []contracts.ArtifactRef{{
		Namespace: SkillNamespace, Name: "review",
	}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = catalog.InitializeRun(ctx, "owner-1", "run-media", selected, [][]string{{"review"}})
	assertRunSkillError(t, err, CodeMediaTypeInvalid, "review", false)
}

func TestRunSkillAggregateLimitsAreDeterministic(t *testing.T) {
	selected := make([]contracts.RunSkillSnapshot, 17)
	names := make([]string, len(selected))
	for index := range selected {
		name := fmt.Sprintf("skill-%02d", index)
		revision := fmt.Sprintf("revision-%02d", index)
		digest := fmt.Sprintf("sha256:%064x", index+1)
		selected[index] = contracts.RunSkillSnapshot{
			Name: name,
			Source: &contracts.ArtifactRef{
				Namespace: SkillNamespace, Name: name, Revision: &revision,
			},
			SourceDigest: digest,
			SourceSize:   MaximumArchiveBytes,
		}
		names[index] = name
	}
	if err := selected[0].Validate(); err != nil {
		t.Fatalf("test snapshot is invalid: %v", err)
	}
	err := ValidateSelectedLimits(selected, [][]string{names[:4], names[4:8]})
	assertRunSkillError(t, err, CodeLimitExceeded, names[16], false)

	selected = selected[:5]
	err = ValidateSelectedLimits(selected, [][]string{names[:5]})
	assertRunSkillError(t, err, CodeLimitExceeded, names[4], false)
}

func canonicalTestPackage(t *testing.T, name, description string) []byte {
	t.Helper()
	plan := bundledPlan(t, map[string]string{name: description})
	return append([]byte(nil), plan.packages[0].payload...)
}

func assertRunSkillError(t *testing.T, err error, code, name string, retryable bool) {
	t.Helper()
	var skillErr *RunSkillError
	if !errors.As(err, &skillErr) || skillErr.Code != code || skillErr.Name != name ||
		skillErr.Retryable != retryable {
		t.Fatalf("Run Skill error = %#v (%v), want code=%s name=%s retryable=%t", skillErr, err, code, name, retryable)
	}
}
