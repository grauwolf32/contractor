package public

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type fakeRuntimeAgentPrincipalManagement struct {
	mu         sync.Mutex
	principals map[string]controlplane.RuntimeAgentPrincipalProjection
	replaceErr error
	// mutations remembers each key's request and committed result, as the
	// real manager's Runtime management receipts do.
	mutations map[string]fakePrincipalMutation
}

// fakePrincipalMutation is one remembered principal mutation: the request
// the key is bound to and the principal it returned (nil for a delete).
type fakePrincipalMutation struct {
	request   string
	principal *controlplane.RuntimeAgentPrincipalProjection
}

func newFakeRuntimeAgentPrincipalManagement() *fakeRuntimeAgentPrincipalManagement {
	return &fakeRuntimeAgentPrincipalManagement{
		principals: make(map[string]controlplane.RuntimeAgentPrincipalProjection),
		mutations:  make(map[string]fakePrincipalMutation),
	}
}

// replayMutation answers a key already used by a committed mutation, before
// any other check, as agentprincipal.Service does: the same request
// replays its stored result, another request with the key is a permanent
// idempotency conflict. Refused mutations are not remembered.
func (f *fakeRuntimeAgentPrincipalManagement) replayMutation(
	key, request string,
) (fakePrincipalMutation, bool, error) {
	mutation, ok := f.mutations[key]
	if !ok {
		return fakePrincipalMutation{}, false, nil
	}
	if mutation.request != request {
		return fakePrincipalMutation{}, false, fmt.Errorf("%w: Runtime Agent management key", contracts.ErrIdempotencyConflict)
	}
	return mutation, true, nil
}

func (f *fakeRuntimeAgentPrincipalManagement) List(
	_ context.Context, after string, limit int,
) ([]controlplane.RuntimeAgentPrincipalProjection, bool, error) {
	// The real service takes the page size and reads the look-ahead row itself.
	if !pagelimit.ValidPage(limit) {
		return nil, false, fmt.Errorf("%w: fake principal page of %d rows", runtimeconfig.ErrInvalid, limit)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	ids := make([]string, 0, len(f.principals))
	for id := range f.principals {
		if id > after {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	hasMore := len(ids) > limit
	if hasMore {
		ids = ids[:limit]
	}
	result := make([]controlplane.RuntimeAgentPrincipalProjection, len(ids))
	for index, id := range ids {
		result[index] = clonePrincipalProjection(f.principals[id])
	}
	return result, hasMore, nil
}

func (f *fakeRuntimeAgentPrincipalManagement) Get(
	_ context.Context, id string,
) (controlplane.RuntimeAgentPrincipalProjection, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	value, ok := f.principals[id]
	if !ok {
		return controlplane.RuntimeAgentPrincipalProjection{}, runtimeconfig.ErrNotFound
	}
	return clonePrincipalProjection(value), nil
}

func (f *fakeRuntimeAgentPrincipalManagement) ReplaceLabels(
	_ context.Context, id string, expected uint64, labels []string, key, actor string, at time.Time,
) (controlplane.RuntimeAgentPrincipalProjection, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	request := fmt.Sprint("replace", id, expected, labels)
	if replay, ok, err := f.replayMutation(key, request); ok || err != nil {
		if err != nil {
			return controlplane.RuntimeAgentPrincipalProjection{}, false, err
		}
		// controlplane.PrincipalOperations reads the current principal after a
		// replay, so replaying the labels of a deleted principal is not found.
		if _, exists := f.principals[id]; !exists {
			return controlplane.RuntimeAgentPrincipalProjection{}, false, runtimeconfig.ErrNotFound
		}
		return clonePrincipalProjection(*replay.principal), true, nil
	}
	value, ok := f.principals[id]
	if !ok {
		return controlplane.RuntimeAgentPrincipalProjection{}, false, runtimeconfig.ErrNotFound
	}
	if f.replaceErr != nil {
		return controlplane.RuntimeAgentPrincipalProjection{}, false, f.replaceErr
	}
	if value.Principal.LabelRevision != expected {
		return controlplane.RuntimeAgentPrincipalProjection{}, false, runtimeconfig.ErrPrecondition
	}
	value.Principal.Labels = append([]string{}, labels...)
	value.Principal.LabelRevision++
	value.Principal.UpdatedBy = actor
	value.Principal.UpdatedAt = at
	f.principals[id] = value
	stored := clonePrincipalProjection(value)
	f.mutations[key] = fakePrincipalMutation{request: request, principal: &stored}
	return clonePrincipalProjection(value), false, nil
}

func (f *fakeRuntimeAgentPrincipalManagement) Delete(
	_ context.Context, id string, expected uint64, key, actor string, at time.Time,
) (bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	request := fmt.Sprint("delete", id, expected)
	if _, ok, err := f.replayMutation(key, request); ok || err != nil {
		return ok, err
	}
	value, ok := f.principals[id]
	if !ok {
		return false, runtimeconfig.ErrNotFound
	}
	if value.Principal.LabelRevision != expected {
		return false, runtimeconfig.ErrPrecondition
	}
	if len(value.Principal.Labels) != 0 || value.Live != nil {
		return false, agentprincipal.ErrInUse
	}
	delete(f.principals, id)
	f.mutations[key] = fakePrincipalMutation{request: request}
	return false, nil
}

func clonePrincipalProjection(value controlplane.RuntimeAgentPrincipalProjection) controlplane.RuntimeAgentPrincipalProjection {
	value.Principal.Labels = append([]string{}, value.Principal.Labels...)
	value.RequiredRuntimeAdapters = append([]string{}, value.RequiredRuntimeAdapters...)
	value.MissingRuntimeAdapters = append([]string{}, value.MissingRuntimeAdapters...)
	if value.Live != nil {
		live := *value.Live
		value.Live = &live
	}
	return value
}

type fakeRuntimeConfigManagement struct {
	mu           sync.Mutex
	resolver     runtimeconfig.GatewayResolver
	versions     map[string]runtimeconfig.Version
	bindings     map[string]runtimeconfig.Binding
	publications map[string]runtimeconfig.PublishResult
	mutations    map[string]runtimeconfig.BindingMutationResult
	// mutationRequests binds each key to its request, as the real store's
	// request digest does.
	mutationRequests map[string]string
	publishErr       error
	createErr        error
	rebindErr        error
	deleteErr        error
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
	if f.publishErr != nil {
		return runtimeconfig.PublishResult{}, f.publishErr
	}
	prepared, err := runtimeconfig.PreparePublication(document)
	if err != nil {
		return runtimeconfig.PublishResult{}, err
	}
	if replay, ok := f.publications[key]; ok {
		if replay.Version.Ref.Name != prepared.Name() || replay.Version.Ref.Version != prepared.Version() {
			return runtimeconfig.PublishResult{}, contracts.ErrIdempotencyConflict
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
	if err := fakePageFetch(limit, runtimeconfig.ErrInvalid); err != nil {
		return nil, err
	}
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
	if err := fakePageFetch(limit, runtimeconfig.ErrInvalid); err != nil {
		return nil, err
	}
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
	if f.createErr != nil {
		return runtimeconfig.BindingMutationResult{}, f.createErr
	}
	if replay, ok, err := f.replayMutation(key, fmt.Sprint("create", label, ref)); ok || err != nil {
		return replay, err
	}
	if _, ok := f.bindings[label]; ok {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrPrecondition
	}
	if version, ok := f.versions[runtimeConfigKey(ref.Name, ref.Version)]; !ok || version.Ref != ref {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrVersionNotFound
	}
	binding := runtimeconfig.Binding{
		Label: label, Ref: ref, Revision: 1, CreatedBy: actor, CreatedAt: at, UpdatedBy: actor, UpdatedAt: at,
	}
	f.bindings[label] = binding
	result := runtimeconfig.BindingMutationResult{Binding: &binding}
	f.recordMutation(key, fmt.Sprint("create", label, ref), result)
	return result, nil
}

func (f *fakeRuntimeConfigManagement) Rebind(
	_ context.Context, label string, expected uint64, ref runtimeconfig.Ref, key, actor string, at time.Time,
) (runtimeconfig.BindingMutationResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.rebindErr != nil {
		return runtimeconfig.BindingMutationResult{}, f.rebindErr
	}
	if replay, ok, err := f.replayMutation(key, fmt.Sprint("rebind", label, expected, ref)); ok || err != nil {
		return replay, err
	}
	binding, ok := f.bindings[label]
	if !ok {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrLabelNotFound
	}
	if binding.Revision != expected {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrPrecondition
	}
	if version, ok := f.versions[runtimeConfigKey(ref.Name, ref.Version)]; !ok || version.Ref != ref {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrVersionNotFound
	}
	if binding.Ref != ref {
		binding.Ref = ref
		binding.Revision++
		binding.UpdatedBy, binding.UpdatedAt = actor, at
		f.bindings[label] = binding
	}
	result := runtimeconfig.BindingMutationResult{Binding: &binding}
	f.recordMutation(key, fmt.Sprint("rebind", label, expected, ref), result)
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
	if replay, ok, err := f.replayMutation(key, fmt.Sprint("delete", label, expected)); ok || err != nil {
		return replay, err
	}
	if label == runtimeconfig.DefaultLabel {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrReserved
	}
	binding, ok := f.bindings[label]
	if !ok {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrLabelNotFound
	}
	if binding.Revision != expected {
		return runtimeconfig.BindingMutationResult{}, runtimeconfig.ErrPrecondition
	}
	delete(f.bindings, label)
	result := runtimeconfig.BindingMutationResult{Deleted: true}
	f.recordMutation(key, fmt.Sprint("delete", label, expected), result)
	return result, nil
}

// replayMutation replays a key bound to the same request and rejects a key
// bound to another one, as runtimeconfig.ManagementService does.
func (f *fakeRuntimeConfigManagement) replayMutation(
	key, request string,
) (runtimeconfig.BindingMutationResult, bool, error) {
	replay, ok := f.mutations[key]
	if !ok {
		return runtimeconfig.BindingMutationResult{}, false, nil
	}
	if f.mutationRequests[key] != request {
		return runtimeconfig.BindingMutationResult{}, false, contracts.ErrIdempotencyConflict
	}
	replay.Replayed = true
	return replay, true, nil
}

func (f *fakeRuntimeConfigManagement) recordMutation(
	key, request string, result runtimeconfig.BindingMutationResult,
) {
	if f.mutationRequests == nil {
		f.mutationRequests = make(map[string]string)
	}
	f.mutations[key] = result
	f.mutationRequests[key] = request
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
	if err := fakePageFetch(limit, credentials.ErrRuntimeCredentialInvalid); err != nil {
		return nil, err
	}
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
			return credentials.RuntimeCredentialCreateResult{}, contracts.ErrIdempotencyConflict
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

func (f *fakeRuntimeCredentialManagement) ValidateRuntimeCredentialUse(
	_ context.Context, user credentials.RuntimeCredentialUser, id string, allowedKinds ...string,
) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	item, ok := f.records[id]
	if !ok || !user.MayUse(item) {
		return credentials.ErrRuntimeCredentialNotFound
	}
	for _, allowed := range allowedKinds {
		if string(item.Kind) == allowed {
			return nil
		}
	}
	return credentials.ErrRuntimeCredentialInvalid
}

func (f *fakeRuntimeCredentialManagement) WithCredentialReferences(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

func runtimeConfigKey(name, version string) string { return name + "\x00" + version }

func cloneRuntimeVersion(value runtimeconfig.Version) runtimeconfig.Version {
	value.CanonicalDocument = append([]byte(nil), value.CanonicalDocument...)
	return value
}

var _ RuntimeConfigManagement = (*fakeRuntimeConfigManagement)(nil)
var _ RuntimeCredentialManagement = (*fakeRuntimeCredentialManagement)(nil)
