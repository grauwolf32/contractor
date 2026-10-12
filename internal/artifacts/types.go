package artifacts

import (
	"context"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const MaxPayloadSize = 64 * 1024 * 1024

type ScopeKind string

const (
	ScopeUser    ScopeKind = "user"
	ScopeProject ScopeKind = "project"
	ScopeRun     ScopeKind = "run"
)

// Scope is created only through UserScope, ProjectScope or RunScope;
// ArtifactRef never carries these fields.
type Scope struct {
	kind ScopeKind
	id   string
}

func UserScope(userID string) (Scope, error) { return newScope(ScopeUser, userID) }

func ProjectScope(projectID string) (Scope, error) { return newScope(ScopeProject, projectID) }

func RunScope(runID string) (Scope, error) { return newScope(ScopeRun, runID) }

func newScope(kind ScopeKind, id string) (Scope, error) {
	if kind != ScopeUser && kind != ScopeProject && kind != ScopeRun ||
		strings.TrimSpace(id) == "" || strings.ContainsRune(id, 0) {
		return Scope{}, ErrInvalidScope
	}
	return Scope{kind: kind, id: id}, nil
}

func (s Scope) Kind() ScopeKind { return s.kind }

func (s Scope) ID() string { return s.id }

type ArtifactRef = contracts.ArtifactRef

type Payload struct {
	MediaType string
	Data      []byte
	prepared  *BlobObject
}

type ReadResult struct {
	Ref               ArtifactRef
	Payload           Payload
	BindingCreatedAt  time.Time
	RevisionCreatedAt time.Time
}

// Exact describes the bytes of this read as the shared exact-content
// descriptor.
func (r ReadResult) Exact() contracts.ExactArtifact {
	return contracts.ExactContent(r.Ref, r.Payload.MediaType, r.Payload.Data)
}

type WriteResult struct {
	Ref               ArtifactRef
	MediaType         string
	Size              int64
	BindingCreatedAt  time.Time
	RevisionCreatedAt time.Time
}

type ForkResult struct {
	SourceRef ArtifactRef
	TargetRef ArtifactRef
	MediaType string
	Size      int64
}

// Metadata describes one exact immutable Artifact revision and its relation to
// the current logical binding. It never contains bytes or a physical blob key.
type Metadata struct {
	GitSource *GitSource  `json:"gitSource,omitempty"`
	Ref       ArtifactRef `json:"artifact"`
	MediaType string      `json:"mediaType"`
	Size      int64       `json:"size"`
	// Digest is an internal projection of the immutable blob identity. It is
	// deliberately absent from the public Metadata JSON contract.
	Digest    string    `json:"-"`
	Current   bool      `json:"current"`
	Frozen    bool      `json:"frozen"`
	CreatedAt time.Time `json:"createdAt"`
}

// Exact describes this revision as the shared exact-content descriptor.
func (m Metadata) Exact() contracts.ExactArtifact {
	return contracts.ExactArtifact{
		Ref: m.Ref.Clone(), Digest: m.Digest, MediaType: m.MediaType, SizeBytes: m.Size,
	}
}

type BindingPageQuery struct {
	Namespace        *string
	ExcludeNamespace *string
	// HiddenNamespace is one exact Namespace a trusted presentation layer
	// hides besides the caller's ExcludeNamespace, such as the reserved
	// audit-standards catalog (ART-R25) or contractor-system (ART-R17).
	HiddenNamespace string
	// ExcludeNamespacePrefix is used by trusted presentation layers to hide
	// purpose-managed Project bindings that have their own acceptance API.
	ExcludeNamespacePrefix string
	AfterNamespace         string
	AfterName              string
	Limit                  int
}

// RefPageQuery selects one keyset page of versionless current bindings in
// namespace and name order (ART-R47).
type RefPageQuery struct {
	Namespace *string
	// AfterNamespace and AfterName name the last binding of the previous page
	// and are both empty for the first page. With Namespace set,
	// AfterNamespace must equal it.
	AfterNamespace string
	AfterName      string
	// ModelVisible excludes the bindings artifactpolicy.IsModelHiddenBinding
	// hides. Storage applies the exclusion before Limit, so it never shortens a
	// page.
	ModelVisible bool
	// Limit is 1 through MaxBindingListLimit (LIM-ART-3).
	Limit int
}

// RefPage is one RefPageQuery result. More reports that further bindings
// follow the last ref.
type RefPage struct {
	Refs []ArtifactRef
	More bool
}

type VersionPageQuery struct {
	BeforeCreatedAt *time.Time
	BeforeRevision  string
	Limit           int
}

type LineagePageQuery struct {
	BeforeCreatedAt      *time.Time
	BeforeTargetRevision string
	BeforeSourceRevision string
	BeforeKind           string
	// ExcludeKind lets a trusted presentation layer hide internal staging
	// lineage while retaining correct keyset pagination in the repository.
	ExcludeKind string
	Limit       int
}

type LineageEdge struct {
	Kind          string      `json:"kind"`
	SourceScope   ScopeKind   `json:"sourceScope"`
	SourceScopeID string      `json:"-"`
	Source        ArtifactRef `json:"source"`
	TargetScope   ScopeKind   `json:"targetScope"`
	TargetScopeID string      `json:"-"`
	Target        ArtifactRef `json:"target"`
	CreatedAt     time.Time   `json:"createdAt"`
}

// Lineage kinds name how a fork related its target to its source (ART-R77).
const (
	LineageInputFork            = "input_fork"
	LineageOutputBind           = "output_bind"
	LineageProjectOutputPublish = "project_output_publish"
	LineageAuditImport          = "audit_import"
)

// Fork describes one trusted fork: a new target binding whose first revision
// reuses the immutable version of a source revision, without copying bytes,
// and a lineage edge of Kind from that source revision to it (ART-R77). The
// purpose-specific Server flow that performs a fork owns its Namespaces, names
// and authorization (ART-R72); Service.Fork enforces only the rules every fork
// shares.
type Fork struct {
	// Kind is one of the Lineage kinds.
	Kind string
	// Source is the scope SourceRef is resolved in. A fork reads a UserScope
	// or ProjectScope source into a RunScope, a RunScope source into a
	// ProjectScope that owns the Run, or one Run binding into another of the
	// same RunScope.
	Source Scope
	// SourceRef is exact, or versionless to fork the current revision.
	SourceRef ArtifactRef
	// SourceFrozen admits only the frozen current revision of the source
	// binding.
	SourceFrozen bool
	Target       Scope
	// TargetRef is versionless. A fork only creates: an existing target
	// binding is a *ConflictError.
	TargetRef ArtifactRef
	// Frozen creates the target binding frozen.
	Frozen bool
	// Replay returns an existing target binding instead of a conflict when its
	// current revision was forked with Kind from the same exact source
	// revision, so a retried fork succeeds.
	Replay bool
}

// Repository is the one storage port of Service. PostgresRepository is its
// only production implementation; the port exists so that the handler and
// catalog unit suites run on in-memory doubles without a database. Service
// validates every request before it reaches the port, so implementations
// trust their arguments.
type Repository interface {
	// Write creates target when expectedRevision is nil and otherwise advances
	// the unfrozen binding whose current revision it is (WritePrecondition).
	Write(context.Context, Scope, ArtifactRef, Payload, *string) (WriteResult, error)
	// CreateFrozen creates target frozen. An existing binding is a
	// *ConflictError or ErrArtifactFrozen, as for a Write that only creates.
	CreateFrozen(context.Context, Scope, ArtifactRef, Payload) (WriteResult, error)
	Read(context.Context, Scope, ArtifactRef) (ReadResult, error)
	Metadata(context.Context, Scope, ArtifactRef) (Metadata, error)
	ListRefs(context.Context, Scope, RefPageQuery) (RefPage, error)
	ListMetadata(context.Context, Scope, BindingPageQuery) ([]Metadata, error)
	ListVersions(context.Context, Scope, ArtifactRef, VersionPageQuery) ([]Metadata, error)
	ListLineage(context.Context, Scope, ArtifactRef, LineagePageQuery) ([]LineageEdge, error)
	ListPrefix(ctx context.Context, scope Scope, namespace, prefix string, limit int) ([]ArtifactRef, error)
	ReadPrefixBatch(ctx context.Context, scope Scope, namespace, prefix string, limit int) ([]ReadResult, error)
	Fork(context.Context, Fork) (ForkResult, error)
	// Freeze freezes every binding of namespace in scope.
	Freeze(ctx context.Context, scope Scope, namespace string) error
	RequireExact(context.Context, Scope, ArtifactRef) error
}

// validateScope rejects a Scope that its constructors would not return, such
// as the zero value.
func validateScope(scope Scope) error {
	_, err := newScope(scope.kind, scope.id)
	return err
}

func validateComponent(value string) error {
	if contracts.ValidateArtifactName(value) != nil {
		return ErrInvalidName
	}
	return nil
}

func validateRef(ref ArtifactRef) error {
	if err := validateComponent(ref.Namespace); err != nil {
		return err
	}
	if err := validateComponent(ref.Name); err != nil {
		return err
	}
	if ref.Revision != nil {
		if err := validateRevision(*ref.Revision); err != nil {
			return err
		}
	}
	return nil
}

// validateTarget validates a versionless write or fork target.
func validateTarget(ref ArtifactRef) error {
	if err := validateRef(ref); err != nil {
		return err
	}
	if ref.Revision != nil {
		return ErrVersionedWriteTarget
	}
	return nil
}

func validateRevision(value string) error {
	if strings.TrimSpace(value) == "" || strings.ContainsRune(value, 0) {
		return ErrInvalidName
	}
	return nil
}

func validateMediaType(value string) error {
	if !contracts.ValidMediaType(value) {
		return ErrInvalidMediaType
	}
	return nil
}

func validatePayload(payload Payload) error {
	if err := validateMediaType(payload.MediaType); err != nil {
		return err
	}
	if len(payload.Data) > MaxPayloadSize {
		return ErrPayloadTooLarge
	}
	return nil
}

func validateExact(ref ArtifactRef) error {
	if err := validateRef(ref); err != nil {
		return err
	}
	if ref.Revision == nil {
		return ErrExactRevisionRequired
	}
	return nil
}
