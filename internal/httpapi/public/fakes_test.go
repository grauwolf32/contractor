package public

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type artifactKey struct {
	kind      artifacts.ScopeKind
	id        string
	namespace string
	name      string
}

type fakeArtifactRepository struct {
	mu         sync.Mutex
	current    map[artifactKey]artifacts.ReadResult
	historical map[artifactKey]map[string]artifacts.ReadResult
	next       int
	writes     int
	reads      int
}

func newFakeArtifactRepository() *fakeArtifactRepository {
	return &fakeArtifactRepository{
		current: make(map[artifactKey]artifacts.ReadResult), historical: make(map[artifactKey]map[string]artifacts.ReadResult),
	}
}

func (f *fakeArtifactRepository) Write(
	_ context.Context,
	scope artifacts.Scope,
	ref artifacts.ArtifactRef,
	payload artifacts.Payload,
	expected *string,
) (artifacts.WriteResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.writes++
	key := artifactKey{scope.Kind(), scope.ID(), ref.Namespace, ref.Name}
	current, exists := f.current[key]
	if !exists && expected != nil || exists && (expected == nil || current.Ref.Revision == nil || *current.Ref.Revision != *expected) {
		return artifacts.WriteResult{}, artifacts.ErrArtifactConflict
	}
	f.next++
	revision := fmt.Sprintf("revision-%d", f.next)
	exact := artifacts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &revision}
	read := artifacts.ReadResult{Ref: exact, Payload: artifacts.Payload{MediaType: payload.MediaType, Data: append([]byte(nil), payload.Data...)}}
	f.current[key] = read
	if f.historical[key] == nil {
		f.historical[key] = make(map[string]artifacts.ReadResult)
	}
	f.historical[key][revision] = read
	return artifacts.WriteResult{Ref: exact, MediaType: payload.MediaType, Size: int64(len(payload.Data))}, nil
}

func (f *fakeArtifactRepository) Read(
	_ context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef,
) (artifacts.ReadResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.reads++
	key := artifactKey{scope.Kind(), scope.ID(), ref.Namespace, ref.Name}
	var result artifacts.ReadResult
	var ok bool
	if ref.Revision == nil {
		result, ok = f.current[key]
	} else {
		result, ok = f.historical[key][*ref.Revision]
	}
	if !ok {
		return artifacts.ReadResult{}, artifacts.ErrArtifactNotFound
	}
	result.Payload.Data = append([]byte(nil), result.Payload.Data...)
	return result, nil
}

func (f *fakeArtifactRepository) List(
	_ context.Context, scope artifacts.Scope, namespace *string,
) ([]artifacts.ArtifactRef, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	result := make([]artifacts.ArtifactRef, 0)
	for key := range f.current {
		if key.kind == scope.Kind() && key.id == scope.ID() && (namespace == nil || key.namespace == *namespace) {
			result = append(result, artifacts.ArtifactRef{Namespace: key.namespace, Name: key.name})
		}
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Namespace == result[j].Namespace {
			return result[i].Name < result[j].Name
		}
		return result[i].Namespace < result[j].Namespace
	})
	return result, nil
}

func (f *fakeArtifactRepository) ForkInput(
	ctx context.Context,
	sourceScope artifacts.Scope,
	source artifacts.ArtifactRef,
	targetScope artifacts.Scope,
	slot string,
) (artifacts.ForkResult, error) {
	read, err := f.Read(ctx, sourceScope, source)
	if err != nil {
		return artifacts.ForkResult{}, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	targetKey := artifactKey{targetScope.Kind(), targetScope.ID(), "inputs", slot}
	if _, exists := f.current[targetKey]; exists {
		return artifacts.ForkResult{}, artifacts.ErrArtifactConflict
	}
	f.next++
	revision := fmt.Sprintf("revision-%d", f.next)
	target := artifacts.ArtifactRef{Namespace: "inputs", Name: slot, Revision: &revision}
	targetRead := artifacts.ReadResult{Ref: target, Payload: read.Payload}
	f.current[targetKey] = targetRead
	f.historical[targetKey] = map[string]artifacts.ReadResult{revision: targetRead}
	return artifacts.ForkResult{
		SourceRef: read.Ref, TargetRef: target, MediaType: read.Payload.MediaType, Size: int64(len(read.Payload.Data)),
	}, nil
}

func (f *fakeArtifactRepository) BindOutputExact(
	ctx context.Context,
	scope artifacts.Scope,
	slot string,
	source artifacts.ArtifactRef,
	_ *string,
) (artifacts.ForkResult, error) {
	read, err := f.Read(ctx, scope, source)
	if err != nil {
		return artifacts.ForkResult{}, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.next++
	revision := fmt.Sprintf("revision-%d", f.next)
	target := artifacts.ArtifactRef{Namespace: "outputs", Name: slot, Revision: &revision}
	key := artifactKey{scope.Kind(), scope.ID(), target.Namespace, target.Name}
	stored := artifacts.ReadResult{Ref: target, Payload: read.Payload}
	f.current[key] = stored
	f.historical[key] = map[string]artifacts.ReadResult{revision: stored}
	return artifacts.ForkResult{SourceRef: read.Ref, TargetRef: target, MediaType: read.Payload.MediaType, Size: int64(len(read.Payload.Data))}, nil
}

func (*fakeArtifactRepository) PinExact(context.Context, artifacts.Scope, artifacts.ArtifactRef, artifacts.PinKind, string) error {
	return nil
}

func (*fakeArtifactRepository) FreezeOutputs(context.Context, artifacts.Scope) error { return nil }

type fakeRunStore struct {
	runs              map[string]runstore.WorkflowRun
	executions        map[string][]runstore.StageExecution
	idempotencyClaims map[string]fakeIdempotencyClaim
}

type fakeIdempotencyClaim struct {
	runID  string
	digest string
}

func newFakeRunStore() *fakeRunStore {
	return &fakeRunStore{
		runs: make(map[string]runstore.WorkflowRun), executions: make(map[string][]runstore.StageExecution),
		idempotencyClaims: make(map[string]fakeIdempotencyClaim),
	}
}

func (f *fakeRunStore) CreateRun(_ context.Context, params runstore.CreateRunParams) (runstore.WorkflowRun, error) {
	if _, exists := f.runs[params.RunID]; exists {
		return runstore.WorkflowRun{}, runstore.ErrConflict
	}
	run := runstore.WorkflowRun{
		RunID: params.RunID, OwnerID: params.OwnerID,
		WorkflowName: params.WorkflowName, WorkflowVersion: params.WorkflowVersion,
		WorkflowSchemaVersion: params.WorkflowSchemaVersion, WorkflowSnapshot: params.WorkflowSnapshot,
		Parameters: params.Parameters, State: runstore.RunInitializing, CreatedAt: time.Now(), UpdatedAt: time.Now(),
	}
	f.runs[params.RunID] = run
	return run, nil
}

func (f *fakeRunStore) CreateRunIdempotent(
	ctx context.Context,
	params runstore.CreateRunIdempotentParams,
) (runstore.WorkflowRun, bool, error) {
	key := params.OwnerID + "\x00" + params.IdempotencyKey
	if existing, ok := f.idempotencyClaims[key]; ok {
		if existing.digest != params.RequestDigest {
			return runstore.WorkflowRun{}, false, runstore.ErrConflict
		}
		run, err := f.GetRun(ctx, existing.runID)
		return run, false, err
	}
	run, err := f.CreateRun(ctx, params.CreateRunParams)
	if err != nil {
		return runstore.WorkflowRun{}, false, err
	}
	f.idempotencyClaims[key] = fakeIdempotencyClaim{runID: run.RunID, digest: params.RequestDigest}
	return run, true, nil
}

func (f *fakeRunStore) GetRun(_ context.Context, runID string) (runstore.WorkflowRun, error) {
	run, ok := f.runs[runID]
	if !ok {
		return runstore.WorkflowRun{}, runstore.ErrNotFound
	}
	return run, nil
}

func (f *fakeRunStore) TransitionRun(
	_ context.Context,
	runID string,
	expected runstore.WorkflowRunState,
	next runstore.WorkflowRunState,
	reason runstore.Reason,
) (runstore.WorkflowRun, error) {
	run, ok := f.runs[runID]
	if !ok {
		return runstore.WorkflowRun{}, runstore.ErrNotFound
	}
	if run.State != expected {
		return runstore.WorkflowRun{}, runstore.ErrConflict
	}
	run.State = next
	run.StateReason = reason
	f.runs[runID] = run
	return run, nil
}

func (f *fakeRunStore) RequestRunCancellation(
	_ context.Context,
	runID string,
	cancellation runstore.WorkflowRunCancellation,
) (runstore.WorkflowRun, error) {
	run, ok := f.runs[runID]
	if !ok {
		return runstore.WorkflowRun{}, runstore.ErrNotFound
	}
	if run.State == runstore.RunInitializing || run.State == runstore.RunRunning {
		version := contracts.APIVersion
		run.State = runstore.RunCancelling
		run.StateReason = runstore.Reason{Code: runstore.CancellationUserRequested}
		run.CancellationSchemaVersion = &version
		run.Cancellation = &cancellation
		f.runs[runID] = run
	}
	return run, nil
}

func (f *fakeRunStore) ListStageExecutions(_ context.Context, runID string) ([]runstore.StageExecution, error) {
	return append([]runstore.StageExecution(nil), f.executions[runID]...), nil
}

type fakeUnitOfWork struct {
	runs      *fakeRunStore
	artifacts *artifacts.Service
	calls     int
}

func (f *fakeUnitOfWork) Do(ctx context.Context, fn func(RunWriter, *artifacts.Service) error) error {
	f.calls++
	return fn(f.runs, f.artifacts)
}

var _ artifacts.Repository = (*fakeArtifactRepository)(nil)
var _ RunReader = (*fakeRunStore)(nil)
var _ RunWriter = (*fakeRunStore)(nil)
var _ UnitOfWork = (*fakeUnitOfWork)(nil)

func exactArtifact(namespace, name, revision string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}
