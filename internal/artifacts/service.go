package artifacts

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/pagelimit"
)

// Service validates every Artifact request once, before it reaches its
// Repository. It exposes only generic operations: the scope views below,
// which apply the generic Namespace policy of the public and private Artifact
// APIs, and the trusted primitives in trusted.go, on which purpose-specific
// Server flows build the operations they own.
type Service struct {
	repository Repository
}

func NewService(repository Repository) *Service { return &Service{repository: repository} }

// ScopedStore is the generic view of one scope. Its writes cannot create or
// advance a binding in a Namespace that a purpose-specific flow owns.
type ScopedStore struct {
	service *Service
	scope   Scope
}

func (s *Service) User(userID string) (ScopedStore, error) {
	scope, err := UserScope(userID)
	if err != nil {
		return ScopedStore{}, err
	}
	return ScopedStore{service: s, scope: scope}, nil
}

func (s *Service) Project(projectID string) (ScopedStore, error) {
	scope, err := ProjectScope(projectID)
	if err != nil {
		return ScopedStore{}, err
	}
	return ScopedStore{service: s, scope: scope}, nil
}

func (s *Service) Run(runID string) (ScopedStore, error) {
	scope, err := RunScope(runID)
	if err != nil {
		return ScopedStore{}, err
	}
	return ScopedStore{service: s, scope: scope}, nil
}

// Scoped returns the view of scope for a trusted caller that already holds
// the Scope of a stored exact Artifact.
func (s *Service) Scoped(scope Scope) (ScopedStore, error) {
	if err := validateScope(scope); err != nil {
		return ScopedStore{}, err
	}
	return ScopedStore{service: s, scope: scope}, nil
}

func (s ScopedStore) Write(
	ctx context.Context,
	target ArtifactRef,
	payload Payload,
	expectedRevision *string,
) (WriteResult, error) {
	if err := validateTarget(target); err != nil {
		return WriteResult{}, err
	}
	if expectedRevision != nil {
		if err := validateRevision(*expectedRevision); err != nil {
			return WriteResult{}, err
		}
	}
	if genericWriteReserved(s.scope, target.Namespace, expectedRevision != nil) {
		return WriteResult{}, ErrReservedNamespace
	}
	if err := validatePayload(payload); err != nil {
		return WriteResult{}, err
	}
	return s.service.repository.Write(ctx, s.scope, target, payload, expectedRevision)
}

// genericWriteReserved reports whether a generic write may not reach
// namespace in scope, because a purpose-specific flow owns it: the Audit
// standard catalog (ART-R25), the Audit-managed Project Namespaces, created
// frozen only by the trusted primitives (ART-R26, REV-R146), and the
// purpose-reserved Run Namespaces (ART-R114). Only the trusted input forks
// create inputs/<slot>; a generic writer may only advance an existing working
// copy by exact CAS (ART-R16, ART-R37, ART-R76).
func genericWriteReserved(scope Scope, namespace string, advancing bool) bool {
	switch scope.kind {
	case ScopeUser:
		return artifactpolicy.IsAuditStandardCatalogNamespace(namespace)
	case ScopeProject:
		return artifactpolicy.IsAuditManagedProjectNamespace(namespace)
	case ScopeRun:
		if namespace == "inputs" {
			return !advancing
		}
		return artifactpolicy.IsPurposeReservedNamespace(namespace)
	}
	return true
}

func (s ScopedStore) Read(ctx context.Context, ref ArtifactRef) (ReadResult, error) {
	if err := validateRef(ref); err != nil {
		return ReadResult{}, err
	}
	return s.service.repository.Read(ctx, s.scope, ref)
}

// ReadExact reads the exact revision expected names and verifies that it
// holds the content expected describes (contracts.ExactArtifact.Verify). An
// invalid descriptor or other content is ErrArtifactIntegrity: stored records
// name exact content, and repeating the read cannot change it.
func (s ScopedStore) ReadExact(ctx context.Context, expected contracts.ExactArtifact) (ReadResult, error) {
	if err := expected.Validate(); err != nil {
		return ReadResult{}, fmt.Errorf("%w: exact artifact descriptor is invalid: %v", ErrArtifactIntegrity, err)
	}
	read, err := s.Read(ctx, expected.Ref)
	if err != nil {
		return ReadResult{}, err
	}
	if err := expected.Verify(read.Exact()); err != nil {
		return ReadResult{}, fmt.Errorf("%w: %w", ErrArtifactIntegrity, err)
	}
	return read, nil
}

func (s ScopedStore) Metadata(ctx context.Context, ref ArtifactRef) (Metadata, error) {
	if err := validateRef(ref); err != nil {
		return Metadata{}, err
	}
	return s.service.repository.Metadata(ctx, s.scope, ref)
}

func (s ScopedStore) ListMetadata(
	ctx context.Context, query BindingPageQuery,
) ([]Metadata, error) {
	if err := validateBindingPageQuery(query); err != nil {
		return nil, err
	}
	return s.service.repository.ListMetadata(ctx, s.scope, query)
}

func (s ScopedStore) ListVersions(
	ctx context.Context, ref ArtifactRef, query VersionPageQuery,
) ([]Metadata, error) {
	if err := validateTarget(ref); err != nil {
		return nil, err
	}
	if err := validateVersionPageQuery(query); err != nil {
		return nil, err
	}
	// Distinguish an unknown binding from an impossible empty history.
	if _, err := s.service.repository.Metadata(ctx, s.scope, ref); err != nil {
		return nil, err
	}
	return s.service.repository.ListVersions(ctx, s.scope, ref, query)
}

func (s ScopedStore) ListLineage(
	ctx context.Context, ref ArtifactRef, query LineagePageQuery,
) ([]LineageEdge, error) {
	if err := validateRef(ref); err != nil {
		return nil, err
	}
	if err := validateLineagePageQuery(query); err != nil {
		return nil, err
	}
	metadata, err := s.service.repository.Metadata(ctx, s.scope, ref)
	if err != nil {
		return nil, err
	}
	return s.service.repository.ListLineage(ctx, s.scope, metadata.Ref, query)
}

func validateBindingPageQuery(query BindingPageQuery) error {
	if !pagelimit.ValidFetch(query.Limit) {
		return ErrInvalidName
	}
	if query.Namespace != nil {
		if err := validateComponent(*query.Namespace); err != nil {
			return err
		}
	}
	if query.ExcludeNamespace != nil {
		if err := validateComponent(*query.ExcludeNamespace); err != nil {
			return err
		}
	}
	if query.HiddenNamespace != "" {
		if err := validateComponent(query.HiddenNamespace); err != nil {
			return err
		}
	}
	if query.ExcludeNamespacePrefix != "" {
		if err := validateComponent(query.ExcludeNamespacePrefix + "x"); err != nil {
			return err
		}
	}
	if (query.AfterNamespace == "") != (query.AfterName == "") {
		return ErrInvalidName
	}
	if query.AfterNamespace != "" {
		if err := validateComponent(query.AfterNamespace); err != nil {
			return err
		}
		if err := validateComponent(query.AfterName); err != nil {
			return err
		}
	}
	return nil
}

func validateVersionPageQuery(query VersionPageQuery) error {
	if !pagelimit.ValidFetch(query.Limit) {
		return ErrInvalidName
	}
	if (query.BeforeCreatedAt == nil) != (query.BeforeRevision == "") {
		return ErrInvalidName
	}
	if query.BeforeCreatedAt != nil {
		if query.BeforeCreatedAt.IsZero() {
			return ErrInvalidName
		}
		return validateRevision(query.BeforeRevision)
	}
	return nil
}

func validateLineagePageQuery(query LineagePageQuery) error {
	if !pagelimit.ValidFetch(query.Limit) {
		return ErrInvalidName
	}
	present := query.BeforeCreatedAt != nil
	if present != (query.BeforeTargetRevision != "") ||
		present != (query.BeforeSourceRevision != "") || present != (query.BeforeKind != "") {
		return ErrInvalidName
	}
	if !present {
		if query.ExcludeKind != "" && !validLineageKind(query.ExcludeKind) {
			return ErrInvalidName
		}
		return nil
	}
	if query.BeforeCreatedAt.IsZero() || validateRevision(query.BeforeTargetRevision) != nil ||
		validateRevision(query.BeforeSourceRevision) != nil ||
		!validLineageKind(query.BeforeKind) || query.ExcludeKind != "" && !validLineageKind(query.ExcludeKind) {
		return ErrInvalidName
	}
	return nil
}

func validLineageKind(value string) bool {
	return value == LineageInputFork || value == LineageOutputBind ||
		value == LineageProjectOutputPublish || value == LineageAuditImport
}
