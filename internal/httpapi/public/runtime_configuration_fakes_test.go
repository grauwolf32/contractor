package public

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type fakeRuntimeConfigManagement struct {
	mu           sync.Mutex
	resolver     runtimeconfig.GatewayResolver
	versions     map[string]runtimeconfig.Version
	bindings     map[string]runtimeconfig.Binding
	publications map[string]runtimeconfig.PublishResult
	mutations    map[string]runtimeconfig.BindingMutationResult
	deleteErr    error
}

func newFakeRuntimeConfigManagement(resolver runtimeconfig.GatewayResolver) *fakeRuntimeConfigManagement {
	builtIn, err := runtimeconfig.DecodeStoredDocument([]byte(runtimeconfig.BuiltInCanonicalDocument))
	if err != nil {
		panic(err)
	}
	builtIn.BuiltIn = true
	builtIn.ActorID = "contractor-bootstrap"
	builtIn.CreatedAt = time.Unix(0, 0).UTC()
	return &fakeRuntimeConfigManagement{
		resolver: resolver,
		versions: map[string]runtimeconfig.Version{runtimeConfigKey(builtIn.Ref.Name, builtIn.Ref.Version): builtIn},
		bindings: map[string]runtimeconfig.Binding{runtimeconfig.DefaultLabel: {
			Label: runtimeconfig.DefaultLabel, Ref: builtIn.Ref, Revision: 1,
			CreatedBy: "contractor-bootstrap", CreatedAt: builtIn.CreatedAt,
			UpdatedBy: "contractor-bootstrap", UpdatedAt: builtIn.CreatedAt,
		}},
		publications: make(map[string]runtimeconfig.PublishResult),
		mutations:    make(map[string]runtimeconfig.BindingMutationResult),
	}
}

func (f *fakeRuntimeConfigManagement) Publish(
	ctx context.Context, document []byte, key, actor string,
) (runtimeconfig.PublishResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	prepared, err := runtimeconfig.PreparePublication(document)
	if err != nil {
		return runtimeconfig.PublishResult{}, err
	}
	if replay, ok := f.publications[key]; ok {
		if replay.Version.Ref.Name != prepared.Name() || replay.Version.Ref.Version != prepared.Version() {
			return runtimeconfig.PublishResult{}, runtimeconfig.ErrConflict
		}
		replay.Replayed = true
		return replay, nil
	}
	version, err := prepared.Resolve(ctx, f.resolver)
	if err != nil {
		return runtimeconfig.PublishResult{}, err
	}
	identity := runtimeConfigKey(version.Ref.Name, version.Ref.Version)
	if existing, ok := f.versions[identity]; ok {
		if existing.Ref != version.Ref {
			return runtimeconfig.PublishResult{}, runtimeconfig.ErrConflict
		}
		version = existing
	} else {
		version.ActorID = actor
		version.CreatedAt = time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
		f.versions[identity] = version
	}
	result := runtimeconfig.PublishResult{Version: version}
	f.publications[key] = result
	return result, nil
}

func (f *fakeRuntimeConfigManagement) ListVersions(
	_ context.Context, afterName, afterVersion string, limit int,
) ([]runtimeconfig.Version, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	items := make([]runtimeconfig.Version, 0, len(f.versions))
	for _, version := range f.versions {
		if runtimeConfigKey(version.Ref.Name, version.Ref.Version) > runtimeConfigKey(afterName, afterVersion) {
			items = append(items, cloneRuntimeVersion(version))
		}
	}
	sort.Slice(items, func(i, j int) bool {
		return runtimeConfigKey(items[i].Ref.Name, items[i].Ref.Version) < runtimeConfigKey(items[j].Ref.Name, items[j].Ref.Version)
	})
	if len(items) > limit {
		items = items[:limit]
	}
	return items, nil
}

func (f *fakeRuntimeConfigManagement) GetVersion(_ context.Context, name, version string) (runtimeconfig.Version, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	value, ok := f.versions[runtimeConfigKey(name, version)]
	if !ok {
		return runtimeconfig.Version{}, runtimeconfig.ErrNotFound
	}
	return cloneRuntimeVersion(value), nil
}

func (f *fakeRuntimeConfigManagement) ListBindings(
	_ context.Context, after string, limit int,
) ([]runtimeconfig.Binding, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	items := make([]runtimeconfig.Binding, 0, len(f.bindings))
	for _, binding := range f.bindings {
		if binding.Label > after {
			items = append(items, binding)
		}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Label < items[j].Label })
	if len(items) > limit {
		items = items[:limit]
	}
	return items, nil
}

func (f *fakeRuntimeConfigManagement) GetBinding(_ context.Context, label string) (runtimeconfig.Binding, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	binding, ok := f.bindings[label]
	if !ok {
		return runtimeconfig.Binding{}, runtimeconfig.ErrNotFound
	}
	return binding, nil
}

func (f *fakeRuntimeConfigManagement) CreateBinding(
	_ context.Context, label string, ref runtimeconfig.Ref, key, actor string, at time.Time,
) (runtimeconfig.BindingMutationResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if replay, ok := f.mutations[key]; ok {
		replay.Replayed = true
		return replay, nil
	}
	if _, ok := f.bindings[label]; ok {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrPrecondition
	}
	if version, ok := f.versions[runtimeConfigKey(ref.Name, ref.Version)]; !ok || version.Ref != ref {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrNotFound
	}
	binding := runtimeconfig.Binding{
		Label: label, Ref: ref, Revision: 1, CreatedBy: actor, CreatedAt: at, UpdatedBy: actor, UpdatedAt: at,
	}
	f.bindings[label] = binding
	result := runtimeconfig.BindingMutationResult{Binding: &binding}
	f.mutations[key] = result
	return result, nil
}

func (f *fakeRuntimeConfigManagement) Rebind(
	_ context.Context, label string, expected uint64, ref runtimeconfig.Ref, key, actor string, at time.Time,
) (runtimeconfig.BindingMutationResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if replay, ok := f.mutations[key]; ok {
		replay.Replayed = true
		return replay, nil
	}
	binding, ok := f.bindings[label]
	if !ok {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrNotFound
	}
	if binding.Revision != expected {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrPrecondition
	}
	if version, ok := f.versions[runtimeConfigKey(ref.Name, ref.Version)]; !ok || version.Ref != ref {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrNotFound
	}
	if binding.Ref != ref {
		binding.Ref = ref
		binding.Revision++
		binding.UpdatedBy, binding.UpdatedAt = actor, at
		f.bindings[label] = binding
	}
	result := runtimeconfig.BindingMutationResult{Binding: &binding}
	f.mutations[key] = result
	return result, nil
}

func (f *fakeRuntimeConfigManagement) DeleteBinding(
	_ context.Context, label string, expected uint64, key, _ string, _ time.Time,
) (runtimeconfig.BindingMutationResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.deleteErr != nil {
		return runtimeconfig.BindingMutationResult{}, f.deleteErr
	}
	if replay, ok := f.mutations[key]; ok {
		replay.Replayed = true
		return replay, nil
	}
	if label == runtimeconfig.DefaultLabel {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrReserved
	}
	binding, ok := f.bindings[label]
	if !ok {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrNotFound
	}
	if binding.Revision != expected {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrPrecondition
	}
	delete(f.bindings, label)
	result := runtimeconfig.BindingMutationResult{Deleted: true}
	f.mutations[key] = result
	return result, nil
}

type fakeRuntimeCredentialManagement struct {
	mu         sync.Mutex
	records    map[string]credentials.RuntimeCredentialMetadata
	creations  map[string]credentials.RuntimeCredentialMetadata
	tombstones map[string]struct{}
	now        time.Time
	deleteErr  error
}

func newFakeRuntimeCredentialManagement() *fakeRuntimeCredentialManagement {
	return &fakeRuntimeCredentialManagement{
		records:    make(map[string]credentials.RuntimeCredentialMetadata),
		creations:  make(map[string]credentials.RuntimeCredentialMetadata),
		tombstones: make(map[string]struct{}),
		now:        time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC),
	}
}

func (f *fakeRuntimeCredentialManagement) List(
	_ context.Context, after string, limit int,
) ([]credentials.RuntimeCredentialMetadata, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	items := make([]credentials.RuntimeCredentialMetadata, 0, len(f.records))
	for _, item := range f.records {
		if item.CredentialID > after {
			items = append(items, item)
		}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].CredentialID < items[j].CredentialID })
	if len(items) > limit {
		items = items[:limit]
	}
	return items, nil
}

func (f *fakeRuntimeCredentialManagement) Get(_ context.Context, id string) (credentials.RuntimeCredentialMetadata, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	item, ok := f.records[id]
	if !ok {
		return credentials.RuntimeCredentialMetadata{}, credentials.ErrRuntimeCredentialNotFound
	}
	return item, nil
}

func (f *fakeRuntimeCredentialManagement) Create(
	_ context.Context, request credentials.RuntimeCredentialCreateRequest,
) (credentials.RuntimeCredentialCreateResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if replay, ok := f.creations[request.IdempotencyKey]; ok {
		if replay.CredentialID != request.CredentialID || replay.Kind != request.Material.Kind() {
			return credentials.RuntimeCredentialCreateResult{}, credentials.ErrRuntimeCredentialConflict
		}
		return credentials.RuntimeCredentialCreateResult{Credential: replay, Replayed: true}, nil
	}
	if _, exists := f.records[request.CredentialID]; exists {
		return credentials.RuntimeCredentialCreateResult{}, credentials.ErrRuntimeCredentialConflict
	}
	metadata := credentials.RuntimeCredentialMetadata{
		CredentialID: request.CredentialID, Kind: request.Material.Kind(), CreatedBy: request.ActorID, CreatedAt: f.now,
	}
	f.records[request.CredentialID] = metadata
	f.creations[request.IdempotencyKey] = metadata
	return credentials.RuntimeCredentialCreateResult{Credential: metadata}, nil
}

func (f *fakeRuntimeCredentialManagement) Delete(
	_ context.Context, id, _ string,
) (credentials.RuntimeCredentialDeleteResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.deleteErr != nil {
		return credentials.RuntimeCredentialDeleteResult{}, f.deleteErr
	}
	if _, deleted := f.tombstones[id]; deleted {
		return credentials.RuntimeCredentialDeleteResult{Replayed: true}, nil
	}
	if _, exists := f.records[id]; !exists {
		return credentials.RuntimeCredentialDeleteResult{}, credentials.ErrRuntimeCredentialNotFound
	}
	delete(f.records, id)
	f.tombstones[id] = struct{}{}
	return credentials.RuntimeCredentialDeleteResult{}, nil
}

func runtimeConfigKey(name, version string) string { return name + "\x00" + version }

func cloneRuntimeVersion(value runtimeconfig.Version) runtimeconfig.Version {
	value.CanonicalDocument = append([]byte(nil), value.CanonicalDocument...)
	return value
}

var _ RuntimeConfigManagement = (*fakeRuntimeConfigManagement)(nil)
var _ RuntimeCredentialManagement = (*fakeRuntimeCredentialManagement)(nil)
