package artifacts

import (
	"context"
	"errors"
	"testing"
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

func exactRef(namespace, name, revision string) ArtifactRef {
	return ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}

type fakeRepository struct {
	writeCalls int
	bindCalls  int
	writeErr   error
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

func (f *fakeRepository) PinExact(context.Context, Scope, ArtifactRef, PinKind, string) error {
	return nil
}

func (f *fakeRepository) FreezeOutputs(context.Context, Scope) error { return nil }
