package artifacts

import (
	"context"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
)

// Service owns validation and creates scope-bound views over one Repository.
type Service struct {
	repository Repository
}

func NewService(repository Repository) *Service { return &Service{repository: repository} }

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

// WriteFindingProposal is the trusted finding-intake write path. The
// namespace is deliberately unavailable through generic allocation Artifact
// writes, while the resulting exact revision remains readable in its RunScope.
func (s *Service) WriteFindingProposal(
	ctx context.Context,
	runID string,
	name string,
	payload Payload,
) (WriteResult, error) {
	scope, err := RunScope(runID)
	if err != nil {
		return WriteResult{}, err
	}
	target := ArtifactRef{Namespace: artifactpolicy.FindingProposalNamespace, Name: name}
	if err := validateRef(target); err != nil {
		return WriteResult{}, err
	}
	if err := validatePayload(payload); err != nil {
		return WriteResult{}, err
	}
	return s.repository.Write(ctx, scope, target, payload, nil)
}

func (s ScopedStore) Write(
	ctx context.Context,
	target ArtifactRef,
	payload Payload,
	expectedRevision *string,
) (WriteResult, error) {
	if err := validateScope(s.scope); err != nil {
		return WriteResult{}, err
	}
	if err := validateRef(target); err != nil {
		return WriteResult{}, err
	}
	if target.Revision != nil {
		return WriteResult{}, ErrVersionedWriteTarget
	}
	if expectedRevision != nil {
		if err := validateRevision(*expectedRevision); err != nil {
			return WriteResult{}, err
		}
	}
	if s.scope.kind == ScopeRun && (target.Namespace == "outputs" || target.Namespace == "skills" ||
		target.Namespace == artifactpolicy.FindingProposalNamespace) {
		return WriteResult{}, ErrReservedNamespace
	}
	if err := validatePayload(payload); err != nil {
		return WriteResult{}, err
	}
	return s.service.repository.Write(ctx, s.scope, target, payload, expectedRevision)
}

type skillForkRepository interface {
	ForkSkill(context.Context, Scope, ArtifactRef, Scope, string) (ForkResult, error)
}

// ForkSkill is a trusted Run-initialization operation. Generic ScopedStore
// writes remain unable to create or advance the reserved skills Namespace.
func (s *Service) ForkSkill(
	ctx context.Context,
	userID string,
	source ArtifactRef,
	runID string,
	name string,
) (ForkResult, error) {
	user, err := UserScope(userID)
	if err != nil {
		return ForkResult{}, err
	}
	run, err := RunScope(runID)
	if err != nil {
		return ForkResult{}, err
	}
	if source.Namespace != "skills" || source.Name != name {
		return ForkResult{}, ErrInvalidName
	}
	if _, err := exactRevision(source); err != nil {
		return ForkResult{}, err
	}
	if err := validateComponent(name); err != nil {
		return ForkResult{}, err
	}
	repository, ok := s.repository.(skillForkRepository)
	if !ok {
		return ForkResult{}, ErrQueryUnsupported
	}
	return repository.ForkSkill(ctx, user, source, run, name)
}

func (s ScopedStore) Read(ctx context.Context, ref ArtifactRef) (ReadResult, error) {
	if err := validateScope(s.scope); err != nil {
		return ReadResult{}, err
	}
	if err := validateRef(ref); err != nil {
		return ReadResult{}, err
	}
	return s.service.repository.Read(ctx, s.scope, ref)
}

func (s ScopedStore) List(ctx context.Context, namespace *string) ([]ArtifactRef, error) {
	if err := validateScope(s.scope); err != nil {
		return nil, err
	}
	if namespace != nil {
		if err := validateComponent(*namespace); err != nil {
			return nil, err
		}
	}
	return s.service.repository.List(ctx, s.scope, namespace)
}

func (s ScopedStore) Metadata(ctx context.Context, ref ArtifactRef) (Metadata, error) {
	if err := validateScope(s.scope); err != nil {
		return Metadata{}, err
	}
	if err := validateRef(ref); err != nil {
		return Metadata{}, err
	}
	repository, ok := s.service.repository.(QueryRepository)
	if !ok {
		return Metadata{}, ErrQueryUnsupported
	}
	return repository.Metadata(ctx, s.scope, ref)
}

func (s ScopedStore) ListMetadata(
	ctx context.Context, query BindingPageQuery,
) ([]Metadata, error) {
	if err := validateScope(s.scope); err != nil {
		return nil, err
	}
	if err := validateBindingPageQuery(query); err != nil {
		return nil, err
	}
	repository, ok := s.service.repository.(QueryRepository)
	if !ok {
		return nil, ErrQueryUnsupported
	}
	return repository.ListMetadata(ctx, s.scope, query)
}

func (s ScopedStore) ListVersions(
	ctx context.Context, ref ArtifactRef, query VersionPageQuery,
) ([]Metadata, error) {
	if err := validateScope(s.scope); err != nil {
		return nil, err
	}
	if err := validateRef(ref); err != nil {
		return nil, err
	}
	if ref.Revision != nil {
		return nil, ErrVersionedWriteTarget
	}
	if err := validateVersionPageQuery(query); err != nil {
		return nil, err
	}
	repository, ok := s.service.repository.(QueryRepository)
	if !ok {
		return nil, ErrQueryUnsupported
	}
	// Distinguish an unknown binding from an impossible empty history.
	if _, err := repository.Metadata(ctx, s.scope, ref); err != nil {
		return nil, err
	}
	return repository.ListVersions(ctx, s.scope, ref, query)
}

func (s ScopedStore) ListLineage(
	ctx context.Context, ref ArtifactRef, query LineagePageQuery,
) ([]LineageEdge, error) {
	if err := validateScope(s.scope); err != nil {
		return nil, err
	}
	if err := validateRef(ref); err != nil {
		return nil, err
	}
	if err := validateLineagePageQuery(query); err != nil {
		return nil, err
	}
	repository, ok := s.service.repository.(QueryRepository)
	if !ok {
		return nil, ErrQueryUnsupported
	}
	metadata, err := repository.Metadata(ctx, s.scope, ref)
	if err != nil {
		return nil, err
	}
	return repository.ListLineage(ctx, s.scope, metadata.Ref, query)
}

func validateBindingPageQuery(query BindingPageQuery) error {
	if query.Limit < 1 || query.Limit > 201 {
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
	if query.Limit < 1 || query.Limit > 201 {
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
	if query.Limit < 1 || query.Limit > 201 {
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

// ForkInput resolves source once in UserScope, creates inputs/<slot> in the
// target RunScope, and records exact lineage without copying payload bytes.
func (s *Service) ForkInput(
	ctx context.Context,
	userID string,
	source ArtifactRef,
	runID string,
	inputSlot string,
) (ForkResult, error) {
	user, err := UserScope(userID)
	if err != nil {
		return ForkResult{}, err
	}
	run, err := RunScope(runID)
	if err != nil {
		return ForkResult{}, err
	}
	if err := validateRef(source); err != nil {
		return ForkResult{}, err
	}
	if err := validateComponent(inputSlot); err != nil {
		return ForkResult{}, err
	}
	return s.repository.ForkInput(ctx, user, source, run, inputSlot)
}

// ForkProjectInput is the trusted Project-Run counterpart of ForkInput. The
// caller must establish Project ownership before selecting the ProjectScope;
// public request bodies never carry a source scope identifier.
func (s *Service) ForkProjectInput(
	ctx context.Context,
	projectID string,
	source ArtifactRef,
	runID string,
	inputSlot string,
) (ForkResult, error) {
	project, err := ProjectScope(projectID)
	if err != nil {
		return ForkResult{}, err
	}
	run, err := RunScope(runID)
	if err != nil {
		return ForkResult{}, err
	}
	if err := validateRef(source); err != nil {
		return ForkResult{}, err
	}
	if err := validateComponent(inputSlot); err != nil {
		return ForkResult{}, err
	}
	return s.repository.ForkInput(ctx, project, source, run, inputSlot)
}

// BindOutputExact is a trusted Scheduler operation. Construct Service with a
// transaction-bound repository when output binding and Run success must be
// committed atomically.
func (s *Service) BindOutputExact(
	ctx context.Context,
	runID string,
	outputSlot string,
	source ArtifactRef,
	expectedOutputRevision *string,
) (ForkResult, error) {
	run, err := RunScope(runID)
	if err != nil {
		return ForkResult{}, err
	}
	if err := validateComponent(outputSlot); err != nil {
		return ForkResult{}, err
	}
	if _, err := exactRevision(source); err != nil {
		return ForkResult{}, err
	}
	if expectedOutputRevision != nil {
		if err := validateRevision(*expectedOutputRevision); err != nil {
			return ForkResult{}, err
		}
	}
	return s.repository.BindOutputExact(ctx, run, outputSlot, source, expectedOutputRevision)
}

type projectOutputPublisher interface {
	PublishRunOutput(context.Context, Scope, ArtifactRef, Scope, string) (ForkResult, error)
}

type auditArtifactImporter interface {
	ImportAuditArtifact(context.Context, Scope, ArtifactRef, Scope, ArtifactRef) (ForkResult, error)
}

type auditArtifactWriter interface {
	WriteAuditArtifact(context.Context, Scope, ArtifactRef, Payload) (WriteResult, error)
}

// PublishRunOutput is a trusted Scheduler operation that creates one
// ProjectScope outputs/<slot> binding from an exact frozen RunScope output.
// It is deliberately create-only: an existing Project binding is a conflict,
// never an implicit replacement.
func (s *Service) PublishRunOutput(
	ctx context.Context,
	runID string,
	projectID string,
	outputSlot string,
	source ArtifactRef,
) (ForkResult, error) {
	run, err := RunScope(runID)
	if err != nil {
		return ForkResult{}, err
	}
	project, err := ProjectScope(projectID)
	if err != nil {
		return ForkResult{}, err
	}
	if err := validateComponent(outputSlot); err != nil {
		return ForkResult{}, err
	}
	if source.Namespace != "outputs" || source.Name != outputSlot {
		return ForkResult{}, ErrInvalidName
	}
	if _, err := exactRevision(source); err != nil {
		return ForkResult{}, err
	}
	repository, ok := s.repository.(projectOutputPublisher)
	if !ok {
		return ForkResult{}, ErrQueryUnsupported
	}
	return repository.PublishRunOutput(ctx, run, source, project, outputSlot)
}

// ImportAuditArtifact is a trusted collection operation. It creates one
// create-only Audit-managed ProjectScope binding backed by an exact immutable
// RunScope revision and records its lineage without copying payload bytes.
// The public Project artifact API reserves audit-* namespaces, so artifacts
// staged before the matching Audit receipt commit are not externally visible.
func (s *Service) ImportAuditArtifact(
	ctx context.Context,
	runID string,
	source ArtifactRef,
	projectID string,
	target ArtifactRef,
) (ForkResult, error) {
	run, err := RunScope(runID)
	if err != nil {
		return ForkResult{}, err
	}
	project, err := ProjectScope(projectID)
	if err != nil {
		return ForkResult{}, err
	}
	if _, err := exactRevision(source); err != nil {
		return ForkResult{}, err
	}
	if target.Revision != nil || !strings.HasPrefix(target.Namespace, "audit-") {
		return ForkResult{}, ErrInvalidName
	}
	if err := validateRef(target); err != nil {
		return ForkResult{}, err
	}
	repository, ok := s.repository.(auditArtifactImporter)
	if !ok {
		return ForkResult{}, ErrQueryUnsupported
	}
	return repository.ImportAuditArtifact(ctx, run, source, project, target)
}

// ImportFindingArtifact is the trusted finding-intake counterpart for an
// ordinary Project Run. Authorization and receipt creation belong to the
// finding service; this method only enforces exact source identity and a
// create-only Audit-managed target in the same Project.
func (s *Service) ImportFindingArtifact(
	ctx context.Context,
	runID string,
	source ArtifactRef,
	projectID string,
	target ArtifactRef,
) (ForkResult, error) {
	return s.ImportAuditArtifact(ctx, runID, source, projectID, target)
}

// WriteAuditArtifact is the trusted counterpart used for Controller-generated
// Audit reports. It creates one frozen, create-only ProjectScope binding in a
// server-reserved audit-* namespace. Public Project artifact routes never
// expose these staging bindings; the Audit report projection exposes them only
// after the matching durable report links have committed.
func (s *Service) WriteAuditArtifact(
	ctx context.Context,
	projectID string,
	target ArtifactRef,
	payload Payload,
) (WriteResult, error) {
	project, err := ProjectScope(projectID)
	if err != nil {
		return WriteResult{}, err
	}
	if target.Revision != nil || !strings.HasPrefix(target.Namespace, "audit-") {
		return WriteResult{}, ErrInvalidName
	}
	if err := validateRef(target); err != nil {
		return WriteResult{}, err
	}
	if err := validatePayload(payload); err != nil {
		return WriteResult{}, err
	}
	repository, ok := s.repository.(auditArtifactWriter)
	if !ok {
		return WriteResult{}, ErrQueryUnsupported
	}
	return repository.WriteAuditArtifact(ctx, project, target, payload)
}

func (s *Service) PinExact(
	ctx context.Context,
	runID string,
	scope Scope,
	ref ArtifactRef,
	kind PinKind,
	pinID string,
) error {
	if _, err := RunScope(runID); err != nil {
		return err
	}
	if err := validateScope(scope); err != nil {
		return err
	}
	if _, err := exactRevision(ref); err != nil {
		return err
	}
	if kind != PinRunInput && kind != PinStageContext && kind != PinStageResult &&
		kind != PinRunOutput && kind != PinFindingProposal && kind != PinFindingEvidence {
		return ErrInvalidName
	}
	if err := validatePinID(pinID); err != nil {
		return err
	}
	return s.repository.PinExact(ctx, runID, scope, ref, kind, pinID)
}

func (s *Service) FreezeRunOutputs(ctx context.Context, runID string) error {
	run, err := RunScope(runID)
	if err != nil {
		return err
	}
	return s.repository.FreezeOutputs(ctx, run)
}
