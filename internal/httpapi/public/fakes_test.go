package public

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type fakeProjectClaim struct {
	projectID string
	digest    string
}

type fakeProjectStore struct {
	mu       sync.Mutex
	projects map[string]projectstore.Project
	claims   map[string]fakeProjectClaim
	nextTime int64
}

func newFakeProjectStore() *fakeProjectStore {
	return &fakeProjectStore{
		projects: make(map[string]projectstore.Project),
		claims:   make(map[string]fakeProjectClaim),
		nextTime: time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC).UnixNano(),
	}
}

func (f *fakeProjectStore) Create(
	_ context.Context, params projectstore.CreateParams,
) (projectstore.Project, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	claimKey := params.OwnerID + "\x00" + params.IdempotencyKey
	if claim, exists := f.claims[claimKey]; exists {
		if claim.digest != params.RequestDigest {
			return projectstore.Project{}, false, projectstore.ErrConflict
		}
		return f.projects[claim.projectID], false, nil
	}
	if !params.Kind.Valid() || strings.TrimSpace(params.Name) == "" ||
		len(params.Name) > projectstore.MaxNameBytes || len(params.Description) > projectstore.MaxDescriptionBytes {
		return projectstore.Project{}, false, projectstore.ErrInvalid
	}
	if _, exists := f.projects[params.ProjectID]; exists {
		return projectstore.Project{}, false, projectstore.ErrConflict
	}
	now := time.Unix(0, f.nextTime).UTC()
	f.nextTime++
	project := projectstore.Project{
		ProjectID: params.ProjectID, OwnerID: params.OwnerID, Kind: params.Kind,
		Name: params.Name, Description: params.Description,
		Lifecycle: projectstore.LifecycleActive, Revision: 1,
		CreatedAt: now, UpdatedAt: now,
	}
	f.projects[params.ProjectID] = project
	f.claims[claimKey] = fakeProjectClaim{projectID: params.ProjectID, digest: params.RequestDigest}
	return project, true, nil
}

func (f *fakeProjectStore) Get(
	_ context.Context, ownerID, projectID string,
) (projectstore.Project, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	project, exists := f.projects[projectID]
	if !exists || project.OwnerID != ownerID {
		return projectstore.Project{}, projectstore.ErrNotFound
	}
	return project, nil
}

func (f *fakeProjectStore) List(
	_ context.Context, params projectstore.ListParams,
) ([]projectstore.Project, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	projects := make([]projectstore.Project, 0)
	for _, project := range f.projects {
		if project.OwnerID != params.OwnerID || params.Kind != nil && project.Kind != *params.Kind {
			continue
		}
		if params.BeforeCreatedAt != nil && !project.CreatedAt.Before(*params.BeforeCreatedAt) &&
			!(project.CreatedAt.Equal(*params.BeforeCreatedAt) && project.ProjectID < params.BeforeProjectID) {
			continue
		}
		projects = append(projects, project)
	}
	sort.Slice(projects, func(i, j int) bool {
		if projects[i].CreatedAt.Equal(projects[j].CreatedAt) {
			return projects[i].ProjectID > projects[j].ProjectID
		}
		return projects[i].CreatedAt.After(projects[j].CreatedAt)
	})
	if len(projects) > params.Limit {
		projects = projects[:params.Limit]
	}
	return projects, nil
}

func (f *fakeProjectStore) Update(
	_ context.Context, params projectstore.UpdateParams,
) (projectstore.Project, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	project, exists := f.projects[params.ProjectID]
	if !exists || project.OwnerID != params.OwnerID {
		return projectstore.Project{}, projectstore.ErrNotFound
	}
	if project.Lifecycle == projectstore.LifecycleDeleting {
		return projectstore.Project{}, projectstore.ErrDeleting
	}
	if project.Revision != params.ExpectedRevision {
		return projectstore.Project{}, projectstore.ErrPrecondition
	}
	if strings.TrimSpace(params.Name) == "" || len(params.Name) > projectstore.MaxNameBytes ||
		len(params.Description) > projectstore.MaxDescriptionBytes {
		return projectstore.Project{}, projectstore.ErrInvalid
	}
	project.Name, project.Description = params.Name, params.Description
	project.HTTPTarget = cloneHTTPOriginTarget(params.HTTPTarget)
	project.Revision++
	project.UpdatedAt = time.Unix(0, f.nextTime).UTC()
	f.nextTime++
	f.projects[project.ProjectID] = project
	return project, nil
}

func (f *fakeProjectStore) BeginDeletion(
	_ context.Context, params projectstore.BeginDeletionParams,
) (projectstore.Project, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	project, exists := f.projects[params.ProjectID]
	if !exists || project.OwnerID != params.OwnerID {
		return projectstore.Project{}, false, projectstore.ErrNotFound
	}
	if project.Lifecycle == projectstore.LifecycleDeleting {
		return project, false, nil
	}
	if project.Revision != params.ExpectedRevision {
		return projectstore.Project{}, false, projectstore.ErrPrecondition
	}
	now := time.Unix(0, f.nextTime).UTC()
	f.nextTime++
	project.Lifecycle = projectstore.LifecycleDeleting
	project.Deletion = &projectstore.Deletion{
		Phase: projectstore.DeletionCancelling, RequestedAt: now,
	}
	project.Revision++
	project.UpdatedAt = now
	f.projects[project.ProjectID] = project
	return project, true, nil
}

type fakeOperationsReader struct {
	mu          sync.Mutex
	snapshot    controlplane.OperationsSnapshot
	changes     []controlplane.OperationsChange
	watchers    map[uint64]chan struct{}
	nextWatcher uint64
}

func newFakeOperationsReader() *fakeOperationsReader {
	return &fakeOperationsReader{snapshot: controlplane.OperationsSnapshot{
		Cursor: controlplane.OperationsCursor{
			Generation: "operations-generation-test",
		},
		RuntimeAgents: []controlplane.RuntimeAgentObservation{},
		Allocations:   []controlplane.AllocationObservation{},
	}, watchers: make(map[uint64]chan struct{})}
}

func (f *fakeOperationsReader) SnapshotOperations() controlplane.OperationsSnapshot {
	f.mu.Lock()
	defer f.mu.Unlock()
	result := f.snapshot
	result.RuntimeAgents = make([]controlplane.RuntimeAgentObservation, len(f.snapshot.RuntimeAgents))
	copy(result.RuntimeAgents, f.snapshot.RuntimeAgents)
	result.Allocations = make([]controlplane.AllocationObservation, len(f.snapshot.Allocations))
	copy(result.Allocations, f.snapshot.Allocations)
	return result
}

func (f *fakeOperationsReader) set(snapshot controlplane.OperationsSnapshot) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.snapshot = snapshot
	f.changes = nil
}

func (f *fakeOperationsReader) InvalidateOperations(
	resource controlplane.OperationsResource,
	resourceID string,
) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.snapshot.Cursor.Revision++
	change := controlplane.OperationsChange{
		Cursor: f.snapshot.Cursor, Resource: resource, ResourceID: resourceID,
		OccurredAt: time.Now().UTC(),
	}
	f.changes = append(f.changes, change)
	for _, watcher := range f.watchers {
		select {
		case watcher <- struct{}{}:
		default:
		}
	}
	return nil
}

func (f *fakeOperationsReader) ReplayOperations(
	after controlplane.OperationsCursor,
) ([]controlplane.OperationsChange, controlplane.OperationsCursor, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if after.Generation != f.snapshot.Cursor.Generation {
		return nil, f.snapshot.Cursor, controlplane.ErrOperationsGeneration
	}
	if after.Revision > f.snapshot.Cursor.Revision {
		return nil, f.snapshot.Cursor, controlplane.ErrOperationsCursor
	}
	result := make([]controlplane.OperationsChange, 0)
	want := after.Revision + 1
	for _, change := range f.changes {
		if change.Cursor.Revision < want {
			continue
		}
		if change.Cursor.Revision != want {
			return nil, f.snapshot.Cursor, controlplane.ErrOperationsGap
		}
		result = append(result, change)
		want++
	}
	if want != f.snapshot.Cursor.Revision+1 {
		return nil, f.snapshot.Cursor, controlplane.ErrOperationsCursor
	}
	return result, f.snapshot.Cursor, nil
}

func (f *fakeOperationsReader) SubscribeOperations() (<-chan struct{}, func()) {
	f.mu.Lock()
	f.nextWatcher++
	id := f.nextWatcher
	updates := make(chan struct{}, 1)
	f.watchers[id] = updates
	f.mu.Unlock()
	var once sync.Once
	return updates, func() {
		once.Do(func() {
			f.mu.Lock()
			delete(f.watchers, id)
			close(updates)
			f.mu.Unlock()
		})
	}
}

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
	created    map[artifactKey]map[string]time.Time
	frozen     map[artifactKey]bool
	lineage    []artifacts.LineageEdge
	queryReads int
}

func newFakeArtifactRepository() *fakeArtifactRepository {
	return &fakeArtifactRepository{
		current: make(map[artifactKey]artifacts.ReadResult), historical: make(map[artifactKey]map[string]artifacts.ReadResult),
		created: make(map[artifactKey]map[string]time.Time), frozen: make(map[artifactKey]bool),
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
	if f.created[key] == nil {
		f.created[key] = make(map[string]time.Time)
	}
	f.created[key][revision] = time.Unix(int64(f.next), 0).UTC()
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
	f.created[targetKey] = map[string]time.Time{revision: time.Unix(int64(f.next), 0).UTC()}
	f.lineage = append(f.lineage, artifacts.LineageEdge{
		Kind:        artifacts.LineageInputFork,
		SourceScope: sourceScope.Kind(), Source: read.Ref,
		TargetScope: targetScope.Kind(), Target: target,
		CreatedAt: time.Unix(int64(f.next), 0).UTC(),
	})
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
	f.created[key] = map[string]time.Time{revision: time.Unix(int64(f.next), 0).UTC()}
	f.lineage = append(f.lineage, artifacts.LineageEdge{
		Kind:        artifacts.LineageOutputBind,
		SourceScope: scope.Kind(), Source: read.Ref,
		TargetScope: scope.Kind(), Target: target,
		CreatedAt: time.Unix(int64(f.next), 0).UTC(),
	})
	return artifacts.ForkResult{SourceRef: read.Ref, TargetRef: target, MediaType: read.Payload.MediaType, Size: int64(len(read.Payload.Data))}, nil
}

func (*fakeArtifactRepository) PinExact(context.Context, string, artifacts.Scope, artifacts.ArtifactRef, artifacts.PinKind, string) error {
	return nil
}

func (f *fakeArtifactRepository) FreezeOutputs(_ context.Context, scope artifacts.Scope) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	for key := range f.current {
		if key.kind == scope.Kind() && key.id == scope.ID() && key.namespace == "outputs" {
			f.frozen[key] = true
		}
	}
	return nil
}

func (f *fakeArtifactRepository) Metadata(
	_ context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef,
) (artifacts.Metadata, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.queryReads++
	return f.metadataLocked(scope, ref)
}

func (f *fakeArtifactRepository) ListMetadata(
	_ context.Context, scope artifacts.Scope, query artifacts.BindingPageQuery,
) ([]artifacts.Metadata, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.queryReads++
	result := make([]artifacts.Metadata, 0)
	for key, read := range f.current {
		if key.kind != scope.Kind() || key.id != scope.ID() ||
			query.Namespace != nil && key.namespace != *query.Namespace ||
			query.ExcludeNamespace != nil && key.namespace == *query.ExcludeNamespace ||
			query.ExcludeNamespacePrefix != "" && strings.HasPrefix(key.namespace, query.ExcludeNamespacePrefix) ||
			query.AfterNamespace != "" && (key.namespace < query.AfterNamespace ||
				key.namespace == query.AfterNamespace && key.name <= query.AfterName) {
			continue
		}
		metadata, _ := f.metadataLocked(scope, read.Ref)
		result = append(result, metadata)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Ref.Namespace == result[j].Ref.Namespace {
			return result[i].Ref.Name < result[j].Ref.Name
		}
		return result[i].Ref.Namespace < result[j].Ref.Namespace
	})
	if len(result) > query.Limit {
		result = result[:query.Limit]
	}
	return result, nil
}

func (f *fakeArtifactRepository) ListVersions(
	_ context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef,
	query artifacts.VersionPageQuery,
) ([]artifacts.Metadata, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.queryReads++
	key := artifactKey{scope.Kind(), scope.ID(), ref.Namespace, ref.Name}
	if _, ok := f.current[key]; !ok {
		return nil, artifacts.ErrArtifactNotFound
	}
	result := make([]artifacts.Metadata, 0, len(f.historical[key]))
	for revision := range f.historical[key] {
		exact := revision
		metadata, _ := f.metadataLocked(scope, artifacts.ArtifactRef{
			Namespace: ref.Namespace, Name: ref.Name, Revision: &exact,
		})
		if query.BeforeCreatedAt != nil && (metadata.CreatedAt.After(*query.BeforeCreatedAt) ||
			metadata.CreatedAt.Equal(*query.BeforeCreatedAt) && revision >= query.BeforeRevision) {
			continue
		}
		result = append(result, metadata)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].CreatedAt.Equal(result[j].CreatedAt) {
			return *result[i].Ref.Revision > *result[j].Ref.Revision
		}
		return result[i].CreatedAt.After(result[j].CreatedAt)
	})
	if len(result) > query.Limit {
		result = result[:query.Limit]
	}
	return result, nil
}

func (f *fakeArtifactRepository) ListLineage(
	_ context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef,
	query artifacts.LineagePageQuery,
) ([]artifacts.LineageEdge, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.queryReads++
	result := make([]artifacts.LineageEdge, 0)
	for _, edge := range f.lineage {
		matchesSource := edge.SourceScope == scope.Kind() && sameExactArtifact(edge.Source, ref)
		matchesTarget := edge.TargetScope == scope.Kind() && sameExactArtifact(edge.Target, ref)
		if !matchesSource && !matchesTarget || edge.Kind == query.ExcludeKind || !fakeLineageBefore(edge, query) {
			continue
		}
		result = append(result, edge)
	}
	sort.Slice(result, func(i, j int) bool { return fakeLineageLess(result[i], result[j]) })
	if len(result) > query.Limit {
		result = result[:query.Limit]
	}
	return result, nil
}

func (f *fakeArtifactRepository) metadataLocked(
	scope artifacts.Scope, ref artifacts.ArtifactRef,
) (artifacts.Metadata, error) {
	key := artifactKey{scope.Kind(), scope.ID(), ref.Namespace, ref.Name}
	read, ok := f.current[key]
	if ref.Revision != nil {
		read, ok = f.historical[key][*ref.Revision]
	}
	if !ok || read.Ref.Revision == nil {
		return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
	}
	current := f.current[key]
	digest := sha256.Sum256(read.Payload.Data)
	return artifacts.Metadata{
		Ref: read.Ref, MediaType: read.Payload.MediaType, Size: int64(len(read.Payload.Data)),
		Digest:  "sha256:" + hex.EncodeToString(digest[:]),
		Current: current.Ref.Revision != nil && *current.Ref.Revision == *read.Ref.Revision,
		Frozen:  f.frozen[key], CreatedAt: f.created[key][*read.Ref.Revision],
	}, nil
}

func sameExactArtifact(left, right artifacts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name &&
		left.Revision != nil && right.Revision != nil && *left.Revision == *right.Revision
}

func fakeLineageBefore(edge artifacts.LineageEdge, query artifacts.LineagePageQuery) bool {
	if query.BeforeCreatedAt == nil {
		return true
	}
	if edge.CreatedAt.Before(*query.BeforeCreatedAt) {
		return true
	}
	if edge.CreatedAt.After(*query.BeforeCreatedAt) {
		return false
	}
	values := []string{*edge.Target.Revision, *edge.Source.Revision, edge.Kind}
	before := []string{query.BeforeTargetRevision, query.BeforeSourceRevision, query.BeforeKind}
	for index := range values {
		if values[index] != before[index] {
			return values[index] < before[index]
		}
	}
	return false
}

func fakeLineageLess(left, right artifacts.LineageEdge) bool {
	if !left.CreatedAt.Equal(right.CreatedAt) {
		return left.CreatedAt.After(right.CreatedAt)
	}
	leftValues := []string{*left.Target.Revision, *left.Source.Revision, left.Kind}
	rightValues := []string{*right.Target.Revision, *right.Source.Revision, right.Kind}
	for index := range leftValues {
		if leftValues[index] != rightValues[index] {
			return leftValues[index] > rightValues[index]
		}
	}
	return false
}

type fakeRunStore struct {
	runs               map[string]runstore.WorkflowRun
	executions         map[string][]runstore.StageExecution
	allocations        map[string][]runstore.StageAllocation
	decisions          map[string][]runstore.StageTransitionDecision
	idempotencyClaims  map[string]fakeIdempotencyClaim
	eventCursors       map[string]runstore.WorkflowRunEventCursor
	runEvents          map[string][]runstore.WorkflowRunEvent
	outputPublications map[string][]runstore.RunOutputPublication
	queueControls      map[string]runstore.OwnerQueueControl
	projects           *fakeProjectStore
	pinRuntimeLabels   func(context.Context, []string, config.CredentialLookup) (runtimeconfig.RunSnapshot, error)
}

func (f *fakeRunStore) PinRuntimeLabels(
	ctx context.Context, labels []string, credentials config.CredentialLookup,
) (runtimeconfig.RunSnapshot, error) {
	if f.pinRuntimeLabels != nil {
		return f.pinRuntimeLabels(ctx, labels, credentials)
	}
	if len(labels) != 0 {
		return runtimeconfig.RunSnapshot{}, runtimeconfig.ErrNotFound
	}
	return runtimeconfig.BuiltInRunSnapshot(), nil
}

type fakePlannerPlanReader struct {
	plans map[string]planner.PlannerPlanProjection
	err   error
}

func (f *fakePlannerPlanReader) LoadPlan(
	_ context.Context, identity planner.SessionIdentity,
) (planner.PlannerPlanProjection, bool, error) {
	if f.err != nil {
		return planner.PlannerPlanProjection{}, false, f.err
	}
	plan, ok := f.plans[identity.StageExecutionID]
	return plan, ok, nil
}

type fakeIdempotencyClaim struct {
	runID  string
	digest string
}

func newFakeRunStore() *fakeRunStore {
	return &fakeRunStore{
		runs: make(map[string]runstore.WorkflowRun), executions: make(map[string][]runstore.StageExecution),
		allocations:        make(map[string][]runstore.StageAllocation),
		decisions:          make(map[string][]runstore.StageTransitionDecision),
		idempotencyClaims:  make(map[string]fakeIdempotencyClaim),
		eventCursors:       make(map[string]runstore.WorkflowRunEventCursor),
		runEvents:          make(map[string][]runstore.WorkflowRunEvent),
		outputPublications: make(map[string][]runstore.RunOutputPublication),
		queueControls:      make(map[string]runstore.OwnerQueueControl),
	}
}

func (f *fakeRunStore) GetOwnerQueueControl(
	_ context.Context, ownerID string,
) (runstore.OwnerQueueControl, error) {
	control, exists := f.queueControls[ownerID]
	if !exists {
		return runstore.OwnerQueueControl{OwnerID: ownerID}, nil
	}
	return control, nil
}

func (f *fakeRunStore) UpdateOwnerQueueControl(
	_ context.Context, params runstore.UpdateOwnerQueueControlParams,
) (runstore.OwnerQueueControl, error) {
	current, exists := f.queueControls[params.OwnerID]
	if !exists {
		current = runstore.OwnerQueueControl{OwnerID: params.OwnerID}
	}
	if current.Revision != params.ExpectedRevision {
		return runstore.OwnerQueueControl{}, runstore.ErrPrecondition
	}
	if current.Revision == 0 {
		current.Revision = 1
		current.Paused = params.Paused
		current.UpdatedAt = time.Unix(1, 0).UTC()
	} else if current.Paused != params.Paused {
		current.Revision++
		current.Paused = params.Paused
		current.UpdatedAt = current.UpdatedAt.Add(time.Microsecond)
	}
	f.queueControls[params.OwnerID] = current
	return current, nil
}

func (f *fakeRunStore) ListRunOutputPublications(
	_ context.Context, runID string,
) ([]runstore.RunOutputPublication, error) {
	if _, ok := f.runs[runID]; !ok {
		return nil, runstore.ErrNotFound
	}
	return append([]runstore.RunOutputPublication(nil), f.outputPublications[runID]...), nil
}

func (f *fakeRunStore) ListRuns(
	_ context.Context, params runstore.ListRunsParams,
) ([]runstore.WorkflowRunSummary, error) {
	selectors, err := runstore.NormalizeRunMetadataLabelSelectors(params.MetadataLabelSelectors)
	if err != nil {
		return nil, err
	}
	result := make([]runstore.WorkflowRunSummary, 0)
	for _, run := range f.runs {
		if run.OwnerID != params.OwnerID || params.State != nil && run.State != *params.State ||
			params.Lifecycle != nil && !params.Lifecycle.Includes(run.State) ||
			params.ProjectID != nil && (run.ProjectID == nil || *run.ProjectID != *params.ProjectID) {
			continue
		}
		matched := true
		for _, selector := range selectors {
			if run.MetadataLabels[selector.Key] != selector.Value {
				matched = false
				break
			}
		}
		if !matched {
			continue
		}
		if params.BeforeCreatedAt != nil && (run.CreatedAt.After(*params.BeforeCreatedAt) ||
			run.CreatedAt.Equal(*params.BeforeCreatedAt) && run.RunID >= params.BeforeRunID) {
			continue
		}
		result = append(result, runstore.WorkflowRunSummary{
			RunID: run.RunID, ProjectID: run.ProjectID,
			WorkflowName: run.WorkflowName, WorkflowVersion: run.WorkflowVersion,
			MetadataLabels: run.MetadataLabels.Clone(),
			State:          run.State, Deletable: f.runDeletable(run),
			CreatedAt: run.CreatedAt, UpdatedAt: run.UpdatedAt,
			FinishedAt: run.FinishedAt,
		})
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].CreatedAt.Equal(result[j].CreatedAt) {
			return result[i].RunID > result[j].RunID
		}
		return result[i].CreatedAt.After(result[j].CreatedAt)
	})
	if len(result) > params.Limit {
		result = result[:params.Limit]
	}
	return result, nil
}

func (f *fakeRunStore) runDeletable(run runstore.WorkflowRun) bool {
	if !runstore.RunLifecycleTerminal.Includes(run.State) {
		return false
	}
	for _, execution := range f.executions[run.RunID] {
		for _, allocation := range f.allocations[execution.StageExecutionID] {
			if allocation.ReleaseCompletedAt == nil {
				return false
			}
		}
	}
	return true
}

func (f *fakeRunStore) DeleteReleasedTerminalRun(
	_ context.Context, ownerID string, runID string,
) error {
	run, ok := f.runs[runID]
	if !ok || run.OwnerID != ownerID {
		return runstore.ErrNotFound
	}
	if !runstore.RunLifecycleTerminal.Includes(run.State) {
		return &runstore.RunNotDeletableError{
			RunID: runID, Reason: runstore.RunNotTerminal,
		}
	}
	for _, execution := range f.executions[runID] {
		for _, allocation := range f.allocations[execution.StageExecutionID] {
			if allocation.ReleaseCompletedAt == nil {
				return &runstore.RunNotDeletableError{
					RunID: runID, Reason: runstore.RunAllocationReleasePending,
				}
			}
		}
	}
	for _, execution := range f.executions[runID] {
		delete(f.allocations, execution.StageExecutionID)
	}
	delete(f.executions, runID)
	delete(f.decisions, runID)
	delete(f.eventCursors, runID)
	delete(f.runEvents, runID)
	delete(f.outputPublications, runID)
	for key, claim := range f.idempotencyClaims {
		if claim.runID == runID {
			delete(f.idempotencyClaims, key)
		}
	}
	delete(f.runs, runID)
	return nil
}

func (f *fakeRunStore) ListRunQueue(
	ctx context.Context, params runstore.ListRunQueueParams,
) ([]runstore.WorkflowRunQueueItem, error) {
	result := make([]runstore.WorkflowRunQueueItem, 0)
	for _, run := range f.runs {
		if run.OwnerID != params.OwnerID ||
			(run.State != runstore.RunInitializing && run.State != runstore.RunRunning && run.State != runstore.RunCancelling) ||
			params.State != nil && run.State != *params.State {
			continue
		}
		if params.AfterCreatedAt != nil && (run.CreatedAt.Before(*params.AfterCreatedAt) ||
			run.CreatedAt.Equal(*params.AfterCreatedAt) && run.RunID <= params.AfterRunID) {
			continue
		}
		item := runstore.WorkflowRunQueueItem{
			RunID: run.RunID, ProjectID: run.ProjectID,
			WorkflowName: run.WorkflowName, WorkflowVersion: run.WorkflowVersion,
			MetadataLabels: run.MetadataLabels.Clone(), State: run.State,
			EventCursor: runstore.WorkflowRunEventCursor{
				Generation: "events-" + run.RunID,
			},
			CreatedAt: run.CreatedAt, UpdatedAt: run.UpdatedAt,
		}
		if cursor, ok := f.eventCursors[run.RunID]; ok {
			item.EventCursor = cursor
		}
		if run.ProjectID != nil {
			if f.projects == nil {
				continue
			}
			project, err := f.projects.Get(ctx, run.OwnerID, *run.ProjectID)
			if err != nil {
				continue
			}
			item.ProjectName = project.Name
			item.ProjectKind = string(project.Kind)
		}
		if params.Membership != nil {
			switch *params.Membership {
			case runstore.RunQueueStandalone:
				if run.ProjectID != nil {
					continue
				}
			case runstore.RunQueueProject:
				if item.ProjectKind != string(projectstore.KindProject) {
					continue
				}
			case runstore.RunQueueEvaluation:
				if item.ProjectKind != string(projectstore.KindEvaluation) {
					continue
				}
			default:
				return nil, runstore.ErrInvalid
			}
		}
		result = append(result, item)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].CreatedAt.Equal(result[j].CreatedAt) {
			return result[i].RunID < result[j].RunID
		}
		return result[i].CreatedAt.Before(result[j].CreatedAt)
	})
	if len(result) > params.Limit {
		result = result[:params.Limit]
	}
	return result, nil
}

func (f *fakeRunStore) CreateRun(_ context.Context, params runstore.CreateRunParams) (runstore.WorkflowRun, error) {
	if _, exists := f.runs[params.RunID]; exists {
		return runstore.WorkflowRun{}, runstore.ErrConflict
	}
	run := runstore.WorkflowRun{
		RunID: params.RunID, OwnerID: params.OwnerID,
		ProjectID:    params.ProjectID,
		WorkflowName: params.WorkflowName, WorkflowVersion: params.WorkflowVersion,
		WorkflowSchemaVersion: params.WorkflowSchemaVersion, WorkflowSnapshot: params.WorkflowSnapshot,
		Parameters:     cloneParameters(params.Parameters),
		MetadataLabels: params.MetadataLabels.Clone(),
		RuntimeLabels:  params.RuntimeConfig.ExplicitLabels(), RuntimeConfig: params.RuntimeConfig.Clone(),
		ProjectHTTPTarget: cloneHTTPOriginTarget(params.ProjectHTTPTarget),
		State:             runstore.RunInitializing, CreatedAt: time.Now(), UpdatedAt: time.Now(),
	}
	f.runs[params.RunID] = cloneFakeWorkflowRun(run)
	return cloneFakeWorkflowRun(run), nil
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

func (f *fakeRunStore) SetRunSkillSelections(
	_ context.Context,
	runID string,
	skills []contracts.RunSkillSnapshot,
) error {
	run, ok := f.runs[runID]
	if !ok {
		return runstore.ErrNotFound
	}
	if run.State != runstore.RunInitializing || len(run.SkillSnapshot) != 0 {
		return runstore.ErrConflict
	}
	run.SkillSnapshot = append([]contracts.RunSkillSnapshot(nil), skills...)
	run.StateReason = runstore.Reason{Code: runstore.SkillInitializationPendingReason}
	f.runs[runID] = run
	return nil
}

func (f *fakeRunStore) LookupRunIdempotency(
	ctx context.Context, ownerID, idempotencyKey, requestDigest string,
) (runstore.WorkflowRun, bool, error) {
	existing, ok := f.idempotencyClaims[ownerID+"\x00"+idempotencyKey]
	if !ok {
		return runstore.WorkflowRun{}, false, nil
	}
	if existing.digest != requestDigest {
		return runstore.WorkflowRun{}, false, runstore.ErrConflict
	}
	run, err := f.GetRun(ctx, existing.runID)
	return run, err == nil, err
}

func (f *fakeRunStore) GetRun(_ context.Context, runID string) (runstore.WorkflowRun, error) {
	run, ok := f.runs[runID]
	if !ok {
		return runstore.WorkflowRun{}, runstore.ErrNotFound
	}
	return cloneFakeWorkflowRun(run), nil
}

func cloneFakeWorkflowRun(run runstore.WorkflowRun) runstore.WorkflowRun {
	if run.ProjectID != nil {
		projectID := *run.ProjectID
		run.ProjectID = &projectID
	}
	run.ProjectHTTPTarget = cloneHTTPOriginTarget(run.ProjectHTTPTarget)
	run.MetadataLabels = run.MetadataLabels.Clone()
	run.RuntimeLabels = append([]string{}, run.RuntimeLabels...)
	run.Parameters = cloneParameters(run.Parameters)
	return run
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

func (f *fakeRunStore) ListStageAllocations(
	_ context.Context, stageExecutionID string,
) ([]runstore.StageAllocation, error) {
	return append([]runstore.StageAllocation(nil), f.allocations[stageExecutionID]...), nil
}

func (f *fakeRunStore) ListStageTransitionDecisions(
	_ context.Context, runID string,
) ([]runstore.StageTransitionDecision, error) {
	return append([]runstore.StageTransitionDecision(nil), f.decisions[runID]...), nil
}

func (f *fakeRunStore) GetRunEventCursor(
	_ context.Context, runID string,
) (runstore.WorkflowRunEventCursor, error) {
	if _, ok := f.runs[runID]; !ok {
		return runstore.WorkflowRunEventCursor{}, runstore.ErrNotFound
	}
	if cursor, ok := f.eventCursors[runID]; ok {
		return cursor, nil
	}
	return runstore.WorkflowRunEventCursor{Generation: "events-test", Sequence: 0}, nil
}

func (f *fakeRunStore) ListRunEvents(
	_ context.Context,
	runID string,
	afterSequence int64,
	limit int,
) ([]runstore.WorkflowRunEvent, error) {
	result := make([]runstore.WorkflowRunEvent, 0, limit)
	for _, event := range f.runEvents[runID] {
		if event.SequenceNumber <= afterSequence {
			continue
		}
		result = append(result, event)
		if len(result) == limit {
			break
		}
	}
	return result, nil
}

type fakeUnitOfWork struct {
	runs      *fakeRunStore
	artifacts *artifacts.Service
	calls     int
}

type fakeRunSkillInitializer struct {
	runs  *fakeRunStore
	calls int
	err   error
}

func (f *fakeRunSkillInitializer) InitializeRunSkills(
	ctx context.Context,
	runID string,
) (runstore.WorkflowRun, error) {
	f.calls++
	run, err := f.runs.GetRun(ctx, runID)
	if err != nil {
		return runstore.WorkflowRun{}, err
	}
	return run, f.err
}

func (f *fakeUnitOfWork) Do(ctx context.Context, fn func(RunWriter, *artifacts.Service) error) error {
	f.calls++
	return fn(f.runs, f.artifacts)
}

var _ artifacts.Repository = (*fakeArtifactRepository)(nil)
var _ artifacts.QueryRepository = (*fakeArtifactRepository)(nil)
var _ RunReader = (*fakeRunStore)(nil)
var _ RunWriter = (*fakeRunStore)(nil)
var _ UnitOfWork = (*fakeUnitOfWork)(nil)

func exactArtifact(namespace, name, revision string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}
