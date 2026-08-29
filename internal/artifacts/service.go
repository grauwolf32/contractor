package artifacts

import "context"

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

func (s *Service) Run(runID string) (ScopedStore, error) {
	scope, err := RunScope(runID)
	if err != nil {
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
	if s.scope.kind == ScopeRun && target.Namespace == "outputs" {
		return WriteResult{}, ErrReservedNamespace
	}
	if err := validatePayload(payload); err != nil {
		return WriteResult{}, err
	}
	return s.service.repository.Write(ctx, s.scope, target, payload, expectedRevision)
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

func (s *Service) PinExact(
	ctx context.Context,
	scope Scope,
	ref ArtifactRef,
	kind PinKind,
	pinID string,
) error {
	if err := validateScope(scope); err != nil {
		return err
	}
	if _, err := exactRevision(ref); err != nil {
		return err
	}
	if kind != PinRunInput && kind != PinStageContext && kind != PinStageResult && kind != PinRunOutput {
		return ErrInvalidName
	}
	if err := validateComponent(pinID); err != nil {
		return err
	}
	return s.repository.PinExact(ctx, scope, ref, kind, pinID)
}

func (s *Service) FreezeRunOutputs(ctx context.Context, runID string) error {
	run, err := RunScope(runID)
	if err != nil {
		return err
	}
	return s.repository.FreezeOutputs(ctx, run)
}
