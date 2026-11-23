package agentskills

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/apierror"
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
	if err := catalog.RequireRunSources(ctx, "owner-1", selected); err != nil {
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

// A current package above the stored package bound (LIM-SKL-17) is the
// permanent skill_limit_exceeded at selection, never a transient read
// (SKL-R117); one at the bound is selected.
func TestCatalogRunSelectionRejectsAnOversizedCurrentPackage(t *testing.T) {
	ctx := context.Background()
	service := artifacts.NewService(newMemoryArtifactRepository())
	catalog, _ := NewCatalog(service)
	owner, _ := service.User("owner-1")
	for _, size := range []int{MaximumArchiveBytes, MaximumArchiveBytes + 1} {
		name := fmt.Sprintf("size-%d", size)
		if _, err := owner.Write(ctx, artifacts.ArtifactRef{Namespace: SkillNamespace, Name: name},
			artifacts.Payload{MediaType: MediaType, Data: make([]byte, size)}, nil); err != nil {
			t.Fatal(err)
		}
	}
	refs := func(size int) []contracts.ArtifactRef {
		return []contracts.ArtifactRef{{Namespace: SkillNamespace, Name: fmt.Sprintf("size-%d", size)}}
	}
	if selected, err := catalog.SelectRunSources(ctx, "owner-1", refs(MaximumArchiveBytes)); err != nil ||
		len(selected) != 1 || selected[0].SourceSize != MaximumArchiveBytes {
		t.Fatalf("selection at the bound = (%+v, %v)", selected, err)
	}
	_, err := catalog.SelectRunSources(ctx, "owner-1", refs(MaximumArchiveBytes+1))
	assertRunSkillError(t, err, CodeLimitExceeded, fmt.Sprintf("size-%d", MaximumArchiveBytes+1), false)
}

// An exact source revision the store no longer holds is the permanent
// skill_artifact_not_found wherever Run creation or initialization meets it:
// the creation lock, the read and the fork (RUN-R27). Every other store
// failure proves nothing about the revision and stays retryable (SKL-R108).
func TestCatalogRunSourceFailuresAreClassified(t *testing.T) {
	transient := errors.New("storage unavailable")
	for _, test := range []struct {
		name      string
		fail      func(*memoryArtifactRepository, contracts.ArtifactRef)
		create    bool
		code      string
		retryable bool
	}{
		{"creation lock finds the source gone", removeSource, true, CodeArtifactNotFound, false},
		{"initialization read finds the source gone", removeSource, false, CodeArtifactNotFound, false},
		{"fork finds the source gone", func(r *memoryArtifactRepository, _ contracts.ArtifactRef) {
			r.forkErr = fmt.Errorf("fork: %w", artifacts.ErrArtifactNotFound)
		}, false, CodeArtifactNotFound, false},
		{"creation lock fails transiently", func(r *memoryArtifactRepository, _ contracts.ArtifactRef) {
			r.requireExactErr = transient
		}, true, CodeArtifactUnavailable, true},
		{"initialization read fails transiently", func(r *memoryArtifactRepository, _ contracts.ArtifactRef) {
			r.readExistingErr = transient
		}, false, CodeArtifactUnavailable, true},
		{"fork fails transiently", func(r *memoryArtifactRepository, _ contracts.ArtifactRef) {
			r.forkErr = transient
		}, false, CodeForkFailed, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := context.Background()
			repository := newMemoryArtifactRepository()
			service := artifacts.NewService(repository)
			catalog, _ := NewCatalog(service)
			owner, _ := service.User("owner-1")
			if _, err := owner.Write(
				ctx, artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "review"},
				artifacts.Payload{MediaType: MediaType, Data: canonicalTestPackage(t, "review", "Review.")}, nil,
			); err != nil {
				t.Fatal(err)
			}
			selected, err := catalog.SelectRunSources(ctx, "owner-1", []contracts.ArtifactRef{{
				Namespace: SkillNamespace, Name: "review",
			}})
			if err != nil || len(selected) != 1 || selected[0].Source == nil {
				t.Fatalf("selected source = (%+v, %v)", selected, err)
			}
			test.fail(repository, *selected[0].Source)
			if test.create {
				err = catalog.RequireRunSources(ctx, "owner-1", selected)
			} else {
				_, err = catalog.InitializeRun(ctx, "owner-1", "run-1", selected, [][]string{{"review"}})
			}
			assertRunSkillError(t, err, test.code, "review", test.retryable)
		})
	}
}

func removeSource(repository *memoryArtifactRepository, source contracts.ArtifactRef) {
	owner, _ := artifacts.UserScope("owner-1")
	repository.removeRevision(owner, source)
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
	selected := make([]RunSkillSnapshot, 17)
	names := make([]string, len(selected))
	for index := range selected {
		name := fmt.Sprintf("skill-%02d", index)
		revision := fmt.Sprintf("revision-%02d", index)
		digest := fmt.Sprintf("sha256:%064x", index+1)
		selected[index] = RunSkillSnapshot{
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
	// A code that a public operation answers keeps the retryability of its
	// registered Kind (SKL-R117).
	if kind := skillErr.ErrorKind(); kind != apierror.InternalError && (kind.Code() != code || kind.Retryable() != retryable) {
		t.Fatalf("Run Skill error %s answers public code %s retryable=%t", code, kind.Code(), kind.Retryable())
	}
}

// TestRunSkillErrorsAnswerTheirRegisteredPublicCode checks the one mapping
// from a Run Skill code to its public Kind (SKL-R117): the codes that Run
// creation and Audit start return answer themselves, with a message naming
// the logical Skill, while the codes that only Run initialization raises are
// never public answers.
func TestRunSkillErrorsAnswerTheirRegisteredPublicCode(t *testing.T) {
	public := map[string]bool{CodeArtifactNotFound: false, CodeArtifactUnavailable: true, CodeLimitExceeded: false}
	for _, code := range []string{
		CodeArtifactNotFound, CodeArtifactUnavailable, CodeLimitExceeded, CodeMediaTypeInvalid, CodeForkConflict,
		CodeForkFailed, CodeArchiveInvalid, CodePathInvalid, CodeMemberForbidden, CodeManifestInvalid, CodeNameMismatch,
	} {
		classified, ok := apierror.Of(fmt.Errorf("create Run: %w", &RunSkillError{Code: code, Name: "review"}))
		if !ok {
			t.Fatalf("%s is not classified", code)
		}
		kind, message := classified.ErrorKind(), apierror.MessageOf(classified)
		retryable, answered := public[code]
		if answered && (kind.Code() != code || kind.Retryable() != retryable || message != kind.Message()+": skills/review") {
			t.Errorf("%s answers %s retryable=%t %q", code, kind.Code(), kind.Retryable(), message)
		}
		if !answered && (kind != apierror.InternalError || message != kind.Message()) {
			t.Errorf("Run initialization code %s answers %s %q", code, kind.Code(), message)
		}
	}
	unnamed, _ := apierror.Of(&RunSkillError{Code: CodeArtifactUnavailable, Retryable: true})
	if message := apierror.MessageOf(unnamed); message != apierror.SkillArtifactUnavailable.Message() {
		t.Errorf("unnamed Skill failure message = %q", message)
	}
}
