package artifacts

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// The trusted primitives reach every Namespace, including those the generic
// scope views refuse. Each purpose-specific Server flow owns the operation it
// builds on them, together with its Namespace, names and authorization: Run
// creation forks inputs and retains the repeat request, the Scheduler binds,
// freezes and publishes outputs, finding intake writes proposals, the Skill
// and Audit standard catalogs seed and fork packages, and Audits import and
// write their material (ART-R72). The primitives enforce only the rules that
// hold for every caller: Run outputs are bound only by a fork (ART-R73), and an
// Audit-managed Project binding is always created frozen (ART-R26, REV-R146).

// Create creates target in scope at its first revision. It never replaces a
// binding: an existing one is a *ConflictError, or ErrArtifactFrozen when it
// is frozen.
func (s *Service) Create(ctx context.Context, scope Scope, target ArtifactRef, payload Payload) (WriteResult, error) {
	if err := validateCreate(scope, target, payload, false); err != nil {
		return WriteResult{}, err
	}
	return s.repository.Write(ctx, scope, target, payload, nil)
}

// CreateFrozen creates target in scope as a frozen binding, the one writer of
// Server-generated Audit material (REV-R146).
//
// Targets are named by their content or by an immutable identity, so a retry
// after a lost acknowledgement finds its binding already written: an existing
// frozen binding holding the payload's bytes and media type is returned as
// written without staging another blob (REV-R149). Any other existing binding
// is ErrAuditArtifactCollision.
func (s *Service) CreateFrozen(ctx context.Context, scope Scope, target ArtifactRef, payload Payload) (WriteResult, error) {
	if err := validateCreate(scope, target, payload, true); err != nil {
		return WriteResult{}, err
	}
	if written, found, err := s.writtenFrozen(ctx, scope, target, payload); err != nil || found {
		return written, err
	}
	written, err := s.repository.CreateFrozen(ctx, scope, target, payload)
	if !errors.Is(err, ErrArtifactConflict) && !errors.Is(err, ErrArtifactFrozen) {
		return written, err
	}
	// A concurrent writer created the binding after the replay check.
	if written, found, replayErr := s.writtenFrozen(ctx, scope, target, payload); replayErr != nil || found {
		return written, replayErr
	}
	return WriteResult{}, err
}

// writtenFrozen is the replay comparison of CreateFrozen. It reports whether
// target is already bound and returns that binding when it is the frozen
// binding payload would create.
func (s *Service) writtenFrozen(
	ctx context.Context, scope Scope, target ArtifactRef, payload Payload,
) (WriteResult, bool, error) {
	existing, err := s.repository.Metadata(ctx, scope, target)
	if errors.Is(err, ErrArtifactNotFound) {
		return WriteResult{}, false, nil
	}
	if err != nil {
		return WriteResult{}, false, err
	}
	expected := contracts.ExactContent(existing.Ref, payload.MediaType, payload.Data)
	if !existing.Frozen || expected.Verify(existing.Exact()) != nil {
		return WriteResult{}, true, fmt.Errorf("%w: %s/%s", ErrAuditArtifactCollision, target.Namespace, target.Name)
	}
	return WriteResult{
		Ref: existing.Ref, MediaType: existing.MediaType, Size: existing.Size,
		BindingCreatedAt: existing.CreatedAt, RevisionCreatedAt: existing.CreatedAt,
	}, true, nil
}

func validateCreate(scope Scope, target ArtifactRef, payload Payload, frozen bool) error {
	if err := validateScope(scope); err != nil {
		return err
	}
	if err := validateTarget(target); err != nil {
		return err
	}
	if scope.kind == ScopeRun && target.Namespace == "outputs" || !frozen && auditManaged(scope, target) {
		return ErrReservedNamespace
	}
	return validatePayload(payload)
}

// auditManaged reports an Audit-managed Project binding (ART-R26).
func auditManaged(scope Scope, target ArtifactRef) bool {
	return scope.kind == ScopeProject && artifactpolicy.IsAuditManagedProjectNamespace(target.Namespace)
}

// Fork creates fork.TargetRef from the source revision fork describes and
// records its lineage. A fork between a ProjectScope and a RunScope requires
// that the Run belongs to the Project; a missing source revision, or one the
// fork does not admit, is ErrArtifactNotFound.
func (s *Service) Fork(ctx context.Context, fork Fork) (ForkResult, error) {
	if !validLineageKind(fork.Kind) {
		return ForkResult{}, ErrInvalidName
	}
	if err := validateScope(fork.Source); err != nil {
		return ForkResult{}, err
	}
	if err := validateScope(fork.Target); err != nil {
		return ForkResult{}, err
	}
	if !forkScopesAllowed(fork.Source, fork.Target) {
		return ForkResult{}, ErrInvalidScope
	}
	if err := validateRef(fork.SourceRef); err != nil {
		return ForkResult{}, err
	}
	if err := validateTarget(fork.TargetRef); err != nil {
		return ForkResult{}, err
	}
	if !fork.Frozen && auditManaged(fork.Target, fork.TargetRef) {
		return ForkResult{}, ErrReservedNamespace
	}
	return s.repository.Fork(ctx, fork)
}

// forkScopesAllowed admits the forks of ART-R72 and ART-R90: a UserScope or
// ProjectScope source into a RunScope, a RunScope source into a ProjectScope,
// and one RunScope binding into another of the same Run.
func forkScopesAllowed(source, target Scope) bool {
	switch {
	case target.kind == ScopeRun:
		return source.kind == ScopeUser || source.kind == ScopeProject || source == target
	case target.kind == ScopeProject:
		return source.kind == ScopeRun
	}
	return false
}

// Freeze freezes every binding of namespace in scope; a namespace without
// bindings is a successful no-op (ART-R96).
func (s *Service) Freeze(ctx context.Context, scope Scope, namespace string) error {
	if err := validateScope(scope); err != nil {
		return err
	}
	if err := validateComponent(namespace); err != nil {
		return err
	}
	return s.repository.Freeze(ctx, scope, namespace)
}

// RequireExact confirms that an exact revision exists and holds a key-share
// lock on it until the caller's transaction ends, so a concurrent purge cannot
// remove it underneath the caller.
func (s *Service) RequireExact(ctx context.Context, scope Scope, ref ArtifactRef) error {
	if err := validateScope(scope); err != nil {
		return err
	}
	if err := validateExact(ref); err != nil {
		return err
	}
	return s.repository.RequireExact(ctx, scope, ref)
}
