package artifacts

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
)

func TestWriteRejectsDistinctInvalidInputsBeforeRepository(t *testing.T) {
	tests := []struct {
		name    string
		scope   string
		ref     ArtifactRef
		payload Payload
		want    error
	}{
		{
			name: "invalid name", scope: "user",
			ref:     ArtifactRef{Namespace: "projects/source", Name: "archive"},
			payload: Payload{MediaType: "application/zip"}, want: ErrInvalidName,
		},
		{
			name: "space in name", scope: "user",
			ref:     ArtifactRef{Namespace: "projects", Name: "review notes"},
			payload: Payload{MediaType: "text/plain"}, want: ErrInvalidName,
		},
		{
			name: "non-ASCII namespace", scope: "user",
			ref:     ArtifactRef{Namespace: "отчеты", Name: "report"},
			payload: Payload{MediaType: "text/plain"}, want: ErrInvalidName,
		},
		{
			name: "parameterized media type", scope: "user",
			ref:     ArtifactRef{Namespace: "projects", Name: "archive"},
			payload: Payload{MediaType: "application/zip; charset=utf-8"}, want: ErrInvalidMediaType,
		},
		{
			name: "wildcard payload media type", scope: "user",
			ref:     ArtifactRef{Namespace: "projects", Name: "archive"},
			payload: Payload{MediaType: "*/*"}, want: ErrInvalidMediaType,
		},
		{
			name: "oversized", scope: "user",
			ref:     ArtifactRef{Namespace: "projects", Name: "archive"},
			payload: Payload{MediaType: "application/zip", Data: make([]byte, MaxPayloadSize+1)}, want: ErrPayloadTooLarge,
		},
		{
			name: "versioned write target", scope: "user",
			ref:     exactRef("projects", "archive", "revision-1"),
			payload: Payload{MediaType: "application/zip"}, want: ErrVersionedWriteTarget,
		},
		{
			name: "reserved Run output", scope: "run",
			ref:     ArtifactRef{Namespace: "outputs", Name: "archive"},
			payload: Payload{MediaType: "application/zip"}, want: ErrReservedNamespace,
		},
		{
			name: "reserved finding proposal", scope: "run",
			ref:     ArtifactRef{Namespace: "finding-proposals", Name: "forged"},
			payload: Payload{MediaType: "application/json"}, want: ErrReservedNamespace,
		},
		{
			name: "reserved Run system record", scope: "run",
			ref: ArtifactRef{
				Namespace: artifactpolicy.RunSystemNamespace,
				Name:      artifactpolicy.RunRepeatRequestName,
			},
			payload: Payload{MediaType: artifactpolicy.RunRepeatRequestMediaType},
			want:    ErrReservedNamespace,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			repository := &fakeRepository{}
			service := NewService(repository)
			var scoped ScopedStore
			var err error
			if test.scope == "run" {
				scoped, err = service.Run("run-1")
			} else {
				scoped, err = service.User("user-1")
			}
			if err != nil {
				t.Fatal(err)
			}
			_, err = scoped.Write(context.Background(), test.ref, test.payload, nil)
			if !errors.Is(err, test.want) {
				t.Fatalf("Write error = %v, want %v", err, test.want)
			}
			if repository.writeCalls != 0 {
				t.Fatalf("repository called %d times for invalid input", repository.writeCalls)
			}
		})
	}
}

func TestWritePropagatesTypedCASConflict(t *testing.T) {
	t.Parallel()

	expected := "old-revision"
	target := ArtifactRef{Namespace: "projects", Name: "source"}
	repository := &fakeRepository{writeErr: &ConflictError{Ref: target, ExpectedRevision: &expected}}
	service := NewService(repository)
	user, _ := service.User("user-1")
	_, err := user.Write(
		context.Background(), target, Payload{MediaType: "text/plain", Data: []byte("new")}, &expected,
	)
	if !errors.Is(err, ErrArtifactConflict) || repository.writeCalls != 1 {
		t.Fatalf("Write = (%v, calls=%d), want ArtifactConflict after one repository call", err, repository.writeCalls)
	}
}

func TestProjectScopeIsDistinctAndDoesNotUseRunReservedNamespaces(t *testing.T) {
	t.Parallel()
	repository := &fakeRepository{}
	service := NewService(repository)
	project, err := service.Project("project-1")
	if err != nil {
		t.Fatal(err)
	}
	if project.scope.Kind() != ScopeProject || project.scope.ID() != "project-1" {
		t.Fatalf("Project scope = (%q, %q)", project.scope.Kind(), project.scope.ID())
	}
	if _, err := project.Write(
		context.Background(), ArtifactRef{Namespace: "outputs", Name: "openapi"},
		Payload{MediaType: "application/yaml", Data: []byte("openapi: 3.1.0")}, nil,
	); err != nil {
		t.Fatalf("Project outputs namespace should be an ordinary writable binding: %v", err)
	}
}

func TestTrustedOperationsRequireExactRefs(t *testing.T) {
	t.Parallel()

	repository := &fakeRepository{}
	service := NewService(repository)
	_, err := service.BindOutputExact(
		context.Background(), "run-1", "result",
		ArtifactRef{Namespace: "builder", Name: "result"}, nil,
	)
	if !errors.Is(err, ErrExactRevisionRequired) || repository.bindCalls != 0 {
		t.Fatalf("BindOutputExact = (%v, calls=%d)", err, repository.bindCalls)
	}
}

func TestBindingPageQueryValidatesExcludedNamespace(t *testing.T) {
	t.Parallel()

	valid := "skills"
	if err := validateBindingPageQuery(BindingPageQuery{
		ExcludeNamespace: &valid,
		Limit:            10,
	}); err != nil {
		t.Fatalf("valid excluded namespace: %v", err)
	}
	invalid := "skills/packages"
	if err := validateBindingPageQuery(BindingPageQuery{
		ExcludeNamespace: &invalid,
		Limit:            10,
	}); !errors.Is(err, ErrInvalidName) {
		t.Fatalf("invalid excluded namespace error = %v, want %v", err, ErrInvalidName)
	}
}

func TestProjectOutputPublicationRequiresMatchingExactRunOutput(t *testing.T) {
	t.Parallel()

	repository := &fakeRepository{}
	service := NewService(repository)
	invalid := []ArtifactRef{
		{Namespace: "outputs", Name: "openapi"},
		exactRef("builder", "openapi", "revision-1"),
		exactRef("outputs", "other", "revision-1"),
	}
	for _, source := range invalid {
		if _, err := service.PublishRunOutput(
			context.Background(), "run-1", "project-1", "openapi", source,
		); err == nil {
			t.Errorf("PublishRunOutput accepted %+v", source)
		}
	}
	if repository.publishCalls != 0 {
		t.Fatalf("invalid publications reached repository %d times", repository.publishCalls)
	}
	source := exactRef("outputs", "openapi", "revision-1")
	if _, err := service.PublishRunOutput(
		context.Background(), "run-1", "project-1", "openapi", source,
	); err != nil || repository.publishCalls != 1 {
		t.Fatalf("valid PublishRunOutput = (%v, calls=%d)", err, repository.publishCalls)
	}
}

func exactRef(namespace, name, revision string) ArtifactRef {
	return ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}

type fakeRepository struct {
	writeCalls   int
	bindCalls    int
	publishCalls int
	writeErr     error
}

func (f *fakeRepository) Write(
	context.Context, Scope, ArtifactRef, Payload, *string,
) (WriteResult, error) {
	f.writeCalls++
	return WriteResult{}, f.writeErr
}

func (f *fakeRepository) Read(context.Context, Scope, ArtifactRef) (ReadResult, error) {
	return ReadResult{}, nil
}

func (f *fakeRepository) List(context.Context, Scope, *string) ([]ArtifactRef, error) {
	return nil, nil
}

func (f *fakeRepository) ForkInput(context.Context, Scope, ArtifactRef, Scope, string) (ForkResult, error) {
	return ForkResult{}, nil
}

func (f *fakeRepository) BindOutputExact(
	context.Context, Scope, string, ArtifactRef, *string,
) (ForkResult, error) {
	f.bindCalls++
	return ForkResult{}, nil
}

func (f *fakeRepository) PublishRunOutput(
	context.Context, Scope, ArtifactRef, Scope, string,
) (ForkResult, error) {
	f.publishCalls++
	return ForkResult{}, nil
}

func (f *fakeRepository) PinExact(context.Context, string, Scope, ArtifactRef, PinKind, string) error {
	return nil
}

func (f *fakeRepository) FreezeOutputs(context.Context, Scope) error { return nil }
