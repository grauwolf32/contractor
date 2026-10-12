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
		{
			// Only the trusted input forks create inputs/<slot> (ART-R16).
			name: "created Run input", scope: "run",
			ref:     ArtifactRef{Namespace: "inputs", Name: "invented"},
			payload: Payload{MediaType: "text/plain"}, want: ErrReservedNamespace,
		},
		{
			name: "reserved Audit standard catalog", scope: "user",
			ref:     ArtifactRef{Namespace: artifactpolicy.AuditStandardCatalogNamespace, Name: "package"},
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

// A Worker advances its working input copy by exact CAS (ART-R37), and a
// UserScope Namespace that only starts with audit-standards stays writable
// (ART-R25).
func TestWriteAllowsInputUpdatesAndSimilarCatalogNames(t *testing.T) {
	t.Parallel()
	repository := &fakeRepository{}
	service := NewService(repository)
	run, _ := service.Run("run-1")
	current := "revision-1"
	if _, err := run.Write(context.Background(), ArtifactRef{Namespace: "inputs", Name: "source"},
		Payload{MediaType: "text/plain", Data: []byte("changed")}, &current); err != nil {
		t.Fatalf("Run input update: %v", err)
	}
	user, _ := service.User("user-1")
	if _, err := user.Write(context.Background(), ArtifactRef{Namespace: "audit-standards-notes", Name: "draft"},
		Payload{MediaType: "text/plain", Data: []byte("notes")}, nil); err != nil {
		t.Fatalf("ordinary UserScope Namespace: %v", err)
	}
	if repository.writeCalls != 2 {
		t.Fatalf("repository writes = %d, want 2", repository.writeCalls)
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
	writes := repository.writeCalls
	if _, err := project.Write(
		context.Background(), ArtifactRef{Namespace: "audit-example", Name: "task"},
		Payload{MediaType: "application/zip", Data: []byte("task")}, nil,
	); !errors.Is(err, ErrReservedNamespace) || repository.writeCalls != writes {
		t.Fatalf("generic Audit namespace write = (%v, calls %d -> %d)", err, writes, repository.writeCalls)
	}
}

// The trusted primitives keep the rules every purpose-specific flow shares
// and reject the rest before storage: Run outputs are bound only by forks,
// Audit-managed Project bindings are created frozen, and forks connect only
// the scopes of ART-R72 and ART-R90.
func TestTrustedPrimitivesValidateBeforeRepository(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	run, otherRun := mustScope(t)(RunScope("run-1")), mustScope(t)(RunScope("run-2"))
	project, user := mustScope(t)(ProjectScope("project-1")), mustScope(t)(UserScope("user-1"))
	payload := Payload{MediaType: "application/json", Data: []byte("{}")}
	source := exactRef("builder", "report", "revision-1")
	notes := ArtifactRef{Namespace: "docs", Name: "notes"}
	input := ArtifactRef{Namespace: "inputs", Name: "spec"}
	output := ArtifactRef{Namespace: "outputs", Name: "report"}
	audit := ArtifactRef{Namespace: "audit-example", Name: "task"}
	create := func(scope Scope, target ArtifactRef, payload Payload, frozen bool) func(*Service) error {
		return func(s *Service) error {
			if frozen {
				_, err := s.CreateFrozen(ctx, scope, target, payload)
				return err
			}
			_, err := s.Create(ctx, scope, target, payload)
			return err
		}
	}
	fork := func(kind string, from, to Scope, target ArtifactRef) Fork {
		return Fork{Kind: kind, Source: from, SourceRef: source, Target: to, TargetRef: target}
	}
	forkCall := func(fork Fork) func(*Service) error {
		return func(s *Service) error {
			_, err := s.Fork(ctx, fork)
			return err
		}
	}
	frozenImport := fork(LineageAuditImport, project, project, audit)
	frozenImport.Frozen = true
	tests := map[string]struct {
		call func(*Service) error
		want error
	}{
		"create in Run outputs":                 {create(run, output, payload, false), ErrReservedNamespace},
		"frozen create in Run outputs":          {create(run, output, payload, true), ErrReservedNamespace},
		"unfrozen create in an Audit Namespace": {create(project, audit, payload, false), ErrReservedNamespace},
		"versioned create target":               {create(user, exactRef("docs", "notes", "revision-1"), payload, false), ErrVersionedWriteTarget},
		"create without a scope":                {create(Scope{}, notes, payload, false), ErrInvalidScope},
		"create with an invalid media type":     {create(user, notes, Payload{MediaType: "*/*"}, false), ErrInvalidMediaType},
		"fork of an unknown kind":               {forkCall(fork("copy", run, run, output)), ErrInvalidName},
		"fork across Runs":                      {forkCall(fork(LineageOutputBind, otherRun, run, output)), ErrInvalidScope},
		"fork into a UserScope":                 {forkCall(fork(LineageInputFork, run, user, input)), ErrInvalidScope},
		"fork between Projects":                 {forkCall(frozenImport), ErrInvalidScope},
		"versioned fork target":                 {forkCall(fork(LineageOutputBind, run, run, source)), ErrVersionedWriteTarget},
		"unfrozen fork into an Audit Namespace": {forkCall(fork(LineageAuditImport, run, project, audit)), ErrReservedNamespace},
		"freeze of an invalid Namespace": {func(s *Service) error {
			return s.Freeze(ctx, run, "outputs/all")
		}, ErrInvalidName},
		"versionless exact requirement": {func(s *Service) error {
			return s.RequireExact(ctx, run, notes)
		}, ErrExactRevisionRequired},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			repository := &fakeRepository{}
			if err := test.call(NewService(repository)); !errors.Is(err, test.want) {
				t.Fatalf("error = %v, want %v", err, test.want)
			}
			if repository.calls() != 0 {
				t.Fatalf("invalid request reached the repository %d times", repository.calls())
			}
		})
	}

	versionless := fork(LineageInputFork, user, run, input)
	versionless.SourceRef = ArtifactRef{Namespace: "docs", Name: "spec"}
	published := fork(LineageProjectOutputPublish, run, project, output)
	published.SourceFrozen = true
	imported := fork(LineageAuditImport, run, project, audit)
	imported.Frozen = true
	repository := &fakeRepository{}
	service := NewService(repository)
	for _, valid := range []Fork{
		versionless, fork(LineageInputFork, project, run, input), fork(LineageOutputBind, run, run, output),
		published, imported,
	} {
		if _, err := service.Fork(ctx, valid); err != nil {
			t.Errorf("valid fork %+v: %v", valid, err)
		}
	}
	if err := create(project, audit, payload, true)(service); err != nil {
		t.Errorf("frozen create in an Audit Namespace: %v", err)
	}
	if repository.forkCalls != 5 || repository.createFrozenCalls != 1 {
		t.Fatalf("valid trusted requests reached the repository %d and %d times",
			repository.forkCalls, repository.createFrozenCalls)
	}
}

func mustScope(t *testing.T) func(Scope, error) Scope {
	return func(scope Scope, err error) Scope {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return scope
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
	if err := validateBindingPageQuery(BindingPageQuery{
		HiddenNamespace: invalid,
		Limit:           10,
	}); !errors.Is(err, ErrInvalidName) {
		t.Fatalf("invalid hidden namespace error = %v, want %v", err, ErrInvalidName)
	}
}

func exactRef(namespace, name, revision string) ArtifactRef {
	return ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}

// fakeRepository counts the requests that reach storage; a test that calls a
// method it does not override panics on the nil Repository.
type fakeRepository struct {
	Repository
	listCalls         int
	writeCalls        int
	forkCalls         int
	createFrozenCalls int
	otherCalls        int
	writeErr          error
}

func (f *fakeRepository) calls() int {
	return f.listCalls + f.writeCalls + f.forkCalls + f.createFrozenCalls + f.otherCalls
}

func (f *fakeRepository) Write(
	context.Context, Scope, ArtifactRef, Payload, *string,
) (WriteResult, error) {
	f.writeCalls++
	return WriteResult{}, f.writeErr
}

func (f *fakeRepository) CreateFrozen(context.Context, Scope, ArtifactRef, Payload) (WriteResult, error) {
	f.createFrozenCalls++
	return WriteResult{}, nil
}

func (f *fakeRepository) Metadata(context.Context, Scope, ArtifactRef) (Metadata, error) {
	return Metadata{}, ErrArtifactNotFound
}

func (f *fakeRepository) ListRefs(context.Context, Scope, RefPageQuery) (RefPage, error) {
	f.listCalls++
	return RefPage{}, nil
}

func (f *fakeRepository) Fork(context.Context, Fork) (ForkResult, error) {
	f.forkCalls++
	return ForkResult{}, nil
}

func (f *fakeRepository) Freeze(context.Context, Scope, string) error {
	f.otherCalls++
	return nil
}

func (f *fakeRepository) RequireExact(context.Context, Scope, ArtifactRef) error {
	f.otherCalls++
	return nil
}

func TestListRefsValidatesThePageBeforeStorage(t *testing.T) {
	repository := &fakeRepository{}
	run, err := NewService(repository).Run("run-1")
	if err != nil {
		t.Fatal(err)
	}
	namespace, other, invalid := "builder", "inputs", "bad/name"
	for name, query := range map[string]RefPageQuery{
		"zero limit":            {Limit: 0},
		"negative limit":        {Limit: -1},
		"limit over maximum":    {Limit: MaxBindingListLimit + 1},
		"invalid namespace":     {Namespace: &invalid, Limit: 1},
		"name without position": {AfterName: "report", Limit: 1},
		"namespace only":        {AfterNamespace: "builder", Limit: 1},
		"invalid position":      {AfterNamespace: "builder", AfterName: "a b", Limit: 1},
		"foreign position":      {Namespace: &namespace, AfterNamespace: other, AfterName: "report", Limit: 1},
	} {
		if _, err := run.ListRefs(context.Background(), query); !errors.Is(err, ErrInvalidName) {
			t.Errorf("%s: error = %v, want ErrInvalidName", name, err)
		}
	}
	if repository.listCalls != 0 {
		t.Fatalf("invalid page queries reached storage %d times", repository.listCalls)
	}
	for _, query := range []RefPageQuery{
		{Limit: 1}, {Limit: MaxBindingListLimit, ModelVisible: true},
		{Namespace: &namespace, AfterNamespace: namespace, AfterName: "report", Limit: 10},
		{AfterNamespace: other, AfterName: "source", Limit: 10},
	} {
		if _, err := run.ListRefs(context.Background(), query); err != nil {
			t.Errorf("valid query %+v: %v", query, err)
		}
	}
	if repository.listCalls != 4 {
		t.Fatalf("valid page queries reached storage %d times", repository.listCalls)
	}
}
