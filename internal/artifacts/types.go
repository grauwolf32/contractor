package artifacts

import (
	"context"
	"regexp"
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
}

type ReadResult struct {
	Ref               ArtifactRef
	Payload           Payload
	BindingCreatedAt  time.Time
	RevisionCreatedAt time.Time
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

type BindingPageQuery struct {
	Namespace        *string
	ExcludeNamespace *string
	// ExcludeNamespacePrefix is used by trusted presentation layers to hide
	// purpose-managed Project bindings that have their own acceptance API.
	ExcludeNamespacePrefix string
	AfterNamespace         string
	AfterName              string
	Limit                  int
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

const (
	LineageInputFork            = "input_fork"
	LineageOutputBind           = "output_bind"
	LineageProjectOutputPublish = "project_output_publish"
	LineageAuditImport          = "audit_import"
)

type PinKind string

const (
	PinRunInput        PinKind = "run_input"
	PinStageContext    PinKind = "stage_context"
	PinStageResult     PinKind = "stage_result"
	PinRunOutput       PinKind = "run_output"
	PinFindingProposal PinKind = "finding_proposal"
	PinFindingEvidence PinKind = "finding_evidence"
)

// Repository is implemented by PostgreSQL and can be bound either to a pool
// or to a caller-owned transaction.
type Repository interface {
	Write(context.Context, Scope, ArtifactRef, Payload, *string) (WriteResult, error)
	Read(context.Context, Scope, ArtifactRef) (ReadResult, error)
	List(context.Context, Scope, *string) ([]ArtifactRef, error)
	ForkInput(context.Context, Scope, ArtifactRef, Scope, string) (ForkResult, error)
	BindOutputExact(context.Context, Scope, string, ArtifactRef, *string) (ForkResult, error)
	PinExact(context.Context, string, Scope, ArtifactRef, PinKind, string) error
	FreezeOutputs(context.Context, Scope) error
}

// QueryRepository is a read-only extension so simple write-path fakes do not
// need to implement UI query semantics. Production PostgreSQL implements both
// interfaces over the same transaction or pool.
type QueryRepository interface {
	Metadata(context.Context, Scope, ArtifactRef) (Metadata, error)
	ListMetadata(context.Context, Scope, BindingPageQuery) ([]Metadata, error)
	ListVersions(context.Context, Scope, ArtifactRef, VersionPageQuery) ([]Metadata, error)
	ListLineage(context.Context, Scope, ArtifactRef, LineagePageQuery) ([]LineageEdge, error)
}

var mediaTypePattern = regexp.MustCompile("^[a-z0-9][a-z0-9!#$%&'+.^_`|~-]*/[a-z0-9][a-z0-9!#$%&'+.^_`|~-]*$")

func validateScope(scope Scope) error {
	if scope.kind != ScopeUser && scope.kind != ScopeProject && scope.kind != ScopeRun ||
		strings.TrimSpace(scope.id) == "" || strings.ContainsRune(scope.id, 0) {
		return ErrInvalidScope
	}
	return nil
}

func validateComponent(value string) error {
	if contracts.ValidateArtifactName(value) != nil {
		return ErrInvalidName
	}
	return nil
}

// Pin IDs are internal opaque identities, not Artifact names.
func validatePinID(value string) error {
	if strings.TrimSpace(value) == "" || strings.Contains(value, "/") || strings.ContainsRune(value, 0) {
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

func validateRevision(value string) error {
	if strings.TrimSpace(value) == "" || strings.ContainsRune(value, 0) {
		return ErrInvalidName
	}
	return nil
}

func validateMediaType(value string) error {
	if value == "*/*" || !mediaTypePattern.MatchString(value) {
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

func exactRevision(ref ArtifactRef) (string, error) {
	if err := validateRef(ref); err != nil {
		return "", err
	}
	if ref.Revision == nil {
		return "", ErrExactRevisionRequired
	}
	return *ref.Revision, nil
}
