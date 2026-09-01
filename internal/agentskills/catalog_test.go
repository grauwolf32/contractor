package agentskills

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestCatalogCreateOnlyInitializationAndDrift(t *testing.T) {
	plan := bundledPlan(t, map[string]string{"beta": "Beta.", "alpha": "Alpha."})
	repository := newMemoryArtifactRepository()
	service := artifacts.NewService(repository)
	catalog, err := NewCatalog(service)
	if err != nil {
		t.Fatal(err)
	}

	first, err := catalog.Initialize(context.Background(), "real-owner", plan)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := outcomeStatuses(first), []string{"alpha:created", "beta:created"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("first outcomes = %v, want %v", got, want)
	}
	second, err := catalog.Initialize(context.Background(), "real-owner", plan)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := outcomeStatuses(second), []string{"alpha:in_sync", "beta:in_sync"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("second outcomes = %v, want %v", got, want)
	}
	if repository.successfulWrites != 2 || repository.revisions != 2 {
		t.Fatalf("writes/revisions = %d/%d, want 2/2", repository.successfulWrites, repository.revisions)
	}
	for key := range repository.bindings {
		if !strings.HasPrefix(key, "user/real-owner/skills/") {
			t.Fatalf("seed escaped real owner UserScope: %q", key)
		}
	}

	owner, _ := service.User("real-owner")
	current, err := owner.Read(context.Background(), artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "alpha"})
	if err != nil {
		t.Fatal(err)
	}
	updated, err := owner.Write(
		context.Background(),
		artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "alpha"},
		artifacts.Payload{MediaType: "application/zip", Data: []byte("operator package")},
		current.Ref.Revision,
	)
	if err != nil {
		t.Fatal(err)
	}
	writesBeforeRestart := repository.successfulWrites
	outcomes, err := catalog.Initialize(context.Background(), "real-owner", plan)
	if err != nil {
		t.Fatal(err)
	}
	if outcomes[0].Status != SeedDrift || outcomes[0].CurrentDigest == outcomes[0].BundledDigest {
		t.Fatalf("drift outcome = %+v", outcomes[0])
	}
	after, err := owner.Read(context.Background(), artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "alpha"})
	if err != nil || after.Ref.Revision == nil || updated.Ref.Revision == nil || *after.Ref.Revision != *updated.Ref.Revision || !bytes.Equal(after.Payload.Data, []byte("operator package")) {
		t.Fatalf("drift was overwritten: result=%+v error=%v", after, err)
	}
	if repository.successfulWrites != writesBeforeRestart {
		t.Fatalf("restart wrote drifted binding: %d -> %d", writesBeforeRestart, repository.successfulWrites)
	}

	wrongMedia, err := owner.Write(
		context.Background(),
		artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "alpha"},
		artifacts.Payload{MediaType: "application/zip", Data: plan.packages[0].payload},
		after.Ref.Revision,
	)
	if err != nil {
		t.Fatal(err)
	}
	outcomes, err = catalog.Initialize(context.Background(), "real-owner", plan)
	if err != nil || outcomes[0].Status != SeedDrift || outcomes[0].CurrentDigest != outcomes[0].BundledDigest {
		t.Fatalf("media-type-only drift = (%+v, %v)", outcomes, err)
	}
	mediaAfter, err := owner.Read(context.Background(), artifacts.ArtifactRef{Namespace: SkillNamespace, Name: "alpha"})
	if err != nil || mediaAfter.Ref.Revision == nil || wrongMedia.Ref.Revision == nil || *mediaAfter.Ref.Revision != *wrongMedia.Ref.Revision {
		t.Fatalf("media drift revision changed: (%+v, %v)", mediaAfter, err)
	}
}

func TestCatalogCrashRecoveryCreatesOnlyMissingBindings(t *testing.T) {
	plan := bundledPlan(t, map[string]string{"alpha": "Alpha.", "beta": "Beta."})
	repository := newMemoryArtifactRepository()
	repository.failAfterSuccessfulWrites = 1
	catalog, _ := NewCatalog(artifacts.NewService(repository))
	if _, err := catalog.Initialize(context.Background(), "owner", plan); err == nil {
		t.Fatal("faulted initialization succeeded")
	}
	if repository.successfulWrites != 1 || repository.revisions != 1 {
		t.Fatalf("partial state = %d writes, %d revisions", repository.successfulWrites, repository.revisions)
	}
	repository.failAfterSuccessfulWrites = -1
	outcomes, err := catalog.Initialize(context.Background(), "owner", plan)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := outcomeStatuses(outcomes), []string{"alpha:in_sync", "beta:created"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("recovery outcomes = %v, want %v", got, want)
	}
	if repository.successfulWrites != 2 || repository.revisions != 2 {
		t.Fatalf("recovery created duplicates: writes=%d revisions=%d", repository.successfulWrites, repository.revisions)
	}
}

func TestCatalogConcurrentCreateCASReconcilesLoser(t *testing.T) {
	plan := bundledPlan(t, map[string]string{"alpha": "Alpha."})
	repository := newMemoryArtifactRepository()
	barrier := &missingReadBarrier{arrived: make(chan struct{}, 2), release: make(chan struct{})}
	repository.missingBarrier = barrier
	catalog, _ := NewCatalog(artifacts.NewService(repository))

	type result struct {
		outcomes []SeedOutcome
		err      error
	}
	results := make(chan result, 2)
	for range 2 {
		go func() {
			outcomes, err := catalog.Initialize(context.Background(), "owner", plan)
			results <- result{outcomes: outcomes, err: err}
		}()
	}
	<-barrier.arrived
	<-barrier.arrived
	close(barrier.release)
	statuses := make([]SeedStatus, 0, 2)
	for range 2 {
		result := <-results
		if result.err != nil || len(result.outcomes) != 1 {
			t.Fatalf("concurrent initialization = (%+v, %v)", result.outcomes, result.err)
		}
		statuses = append(statuses, result.outcomes[0].Status)
	}
	sort.Slice(statuses, func(i, j int) bool { return statuses[i] < statuses[j] })
	if !reflect.DeepEqual(statuses, []SeedStatus{SeedCreated, SeedInSync}) {
		t.Fatalf("concurrent statuses = %v", statuses)
	}
	if repository.successfulWrites != 1 || repository.attemptedWrites != 2 || repository.revisions != 1 {
		t.Fatalf("CAS counters = successful:%d attempted:%d revisions:%d", repository.successfulWrites, repository.attemptedWrites, repository.revisions)
	}
}

func TestBundledDiscoveryIsAllOrNothingAndBounded(t *testing.T) {
	root := t.TempDir()
	if plan, err := DiscoverBundled(root); err != nil || len(plan.Packages()) != 0 {
		t.Fatalf("missing skills subtree = (%+v, %v)", plan, err)
	}
	writeSkillSource(t, root, "valid", "Valid.")
	invalid := filepath.Join(root, SkillNamespace, "invalid")
	if err := os.Mkdir(invalid, 0o755); err != nil {
		t.Fatal(err)
	}
	secret := "DO-NOT-EXPOSE-INSTRUCTION"
	if err := os.WriteFile(filepath.Join(invalid, "SKILL.md"), []byte(secret), 0o644); err != nil {
		t.Fatal(err)
	}
	plan, err := DiscoverBundled(root)
	if plan != nil || ErrorCode(err) != CodeManifestInvalid || strings.Contains(err.Error(), root) || strings.Contains(err.Error(), secret) {
		t.Fatalf("unsafe/non-atomic discovery = (%+v, %v)", plan, err)
	}

	if err := os.RemoveAll(filepath.Join(root, SkillNamespace)); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(root, SkillNamespace), 0o755); err != nil {
		t.Fatal(err)
	}
	for index := 0; index <= MaximumBundledSkills; index++ {
		if err := os.Mkdir(filepath.Join(root, SkillNamespace, fmt.Sprintf("s%03d", index)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := DiscoverBundled(root); ErrorCode(err) != CodeLimitExceeded {
		t.Fatalf("directory count error = %v", err)
	}
}

func bundledPlan(t *testing.T, skills map[string]string) *SeedPlan {
	t.Helper()
	root := t.TempDir()
	for name, description := range skills {
		writeSkillSource(t, root, name, description)
	}
	plan, err := DiscoverBundled(root)
	if err != nil {
		t.Fatal(err)
	}
	metadata := plan.Packages()
	for index := 1; index < len(metadata); index++ {
		if metadata[index-1].Name >= metadata[index].Name {
			t.Fatalf("plan is not sorted: %+v", metadata)
		}
	}
	return plan
}

func writeSkillSource(t *testing.T, operatorRoot, name, description string) {
	t.Helper()
	directory := filepath.Join(operatorRoot, SkillNamespace, name)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	document := []byte("---\nname: " + name + "\ndescription: " + description + "\n---\n# " + name + "\n")
	if err := os.WriteFile(filepath.Join(directory, "SKILL.md"), document, 0o644); err != nil {
		t.Fatal(err)
	}
}

func outcomeStatuses(outcomes []SeedOutcome) []string {
	result := make([]string, len(outcomes))
	for index, outcome := range outcomes {
		result[index] = outcome.Name + ":" + string(outcome.Status)
	}
	return result
}

type memoryArtifact struct {
	result artifacts.ReadResult
}

type missingReadBarrier struct {
	arrived chan struct{}
	release chan struct{}
}

type memoryArtifactRepository struct {
	mu                        sync.Mutex
	bindings                  map[string]memoryArtifact
	history                   map[string]map[string]memoryArtifact
	forkSources               map[string]artifacts.ArtifactRef
	forkCalls                 int
	revisions                 int
	attemptedWrites           int
	successfulWrites          int
	failAfterSuccessfulWrites int
	missingBarrier            *missingReadBarrier
}

func newMemoryArtifactRepository() *memoryArtifactRepository {
	return &memoryArtifactRepository{
		bindings: make(map[string]memoryArtifact), history: make(map[string]map[string]memoryArtifact),
		forkSources: make(map[string]artifacts.ArtifactRef), failAfterSuccessfulWrites: -1,
	}
}

func artifactKey(scope artifacts.Scope, ref artifacts.ArtifactRef) string {
	return string(scope.Kind()) + "/" + scope.ID() + "/" + ref.Namespace + "/" + ref.Name
}

func (r *memoryArtifactRepository) Read(_ context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef) (artifacts.ReadResult, error) {
	r.mu.Lock()
	key := artifactKey(scope, ref)
	stored, exists := r.bindings[key]
	if ref.Revision != nil {
		stored, exists = r.history[key][*ref.Revision]
	}
	barrier := r.missingBarrier
	r.mu.Unlock()
	if !exists {
		if barrier != nil {
			barrier.arrived <- struct{}{}
			<-barrier.release
		}
		return artifacts.ReadResult{}, artifacts.ErrArtifactNotFound
	}
	result := stored.result
	result.Payload.Data = append([]byte(nil), stored.result.Payload.Data...)
	return result, nil
}

func (r *memoryArtifactRepository) Write(_ context.Context, scope artifacts.Scope, target artifacts.ArtifactRef, payload artifacts.Payload, expected *string) (artifacts.WriteResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.attemptedWrites++
	key := artifactKey(scope, target)
	current, exists := r.bindings[key]
	if expected == nil && exists || expected != nil && (!exists || current.result.Ref.Revision == nil || *current.result.Ref.Revision != *expected) {
		return artifacts.WriteResult{}, &artifacts.ConflictError{Ref: target, ExpectedRevision: expected}
	}
	if r.failAfterSuccessfulWrites >= 0 && r.successfulWrites >= r.failAfterSuccessfulWrites {
		return artifacts.WriteResult{}, errors.New("injected write failure")
	}
	r.revisions++
	revision := fmt.Sprintf("rev_%d", r.revisions)
	exact := revision
	now := time.Unix(int64(r.revisions), 0).UTC()
	read := artifacts.ReadResult{
		Ref:              artifacts.ArtifactRef{Namespace: target.Namespace, Name: target.Name, Revision: &exact},
		Payload:          artifacts.Payload{MediaType: payload.MediaType, Data: append([]byte(nil), payload.Data...)},
		BindingCreatedAt: now, RevisionCreatedAt: now,
	}
	r.bindings[key] = memoryArtifact{result: read}
	if r.history[key] == nil {
		r.history[key] = make(map[string]memoryArtifact)
	}
	r.history[key][revision] = memoryArtifact{result: read}
	r.successfulWrites++
	return artifacts.WriteResult{Ref: read.Ref, MediaType: payload.MediaType, Size: int64(len(payload.Data)), BindingCreatedAt: now, RevisionCreatedAt: now}, nil
}

func (r *memoryArtifactRepository) List(context.Context, artifacts.Scope, *string) ([]artifacts.ArtifactRef, error) {
	return nil, artifacts.ErrQueryUnsupported
}
func (r *memoryArtifactRepository) ForkInput(context.Context, artifacts.Scope, artifacts.ArtifactRef, artifacts.Scope, string) (artifacts.ForkResult, error) {
	return artifacts.ForkResult{}, artifacts.ErrQueryUnsupported
}
func (r *memoryArtifactRepository) BindOutputExact(context.Context, artifacts.Scope, string, artifacts.ArtifactRef, *string) (artifacts.ForkResult, error) {
	return artifacts.ForkResult{}, artifacts.ErrQueryUnsupported
}
func (r *memoryArtifactRepository) PinExact(_ context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef, _ artifacts.PinKind, _ string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if ref.Revision == nil {
		return artifacts.ErrExactRevisionRequired
	}
	if _, ok := r.history[artifactKey(scope, ref)][*ref.Revision]; !ok {
		return artifacts.ErrArtifactNotFound
	}
	return nil
}
func (r *memoryArtifactRepository) FreezeOutputs(context.Context, artifacts.Scope) error {
	return artifacts.ErrQueryUnsupported
}

func (r *memoryArtifactRepository) Metadata(_ context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef) (artifacts.Metadata, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := artifactKey(scope, ref)
	stored, ok := r.bindings[key]
	if ref.Revision != nil {
		stored, ok = r.history[key][*ref.Revision]
	}
	if !ok || stored.result.Ref.Revision == nil {
		return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
	}
	digest := sha256.Sum256(stored.result.Payload.Data)
	return artifacts.Metadata{
		Ref: stored.result.Ref, MediaType: stored.result.Payload.MediaType,
		Size: int64(len(stored.result.Payload.Data)), Digest: "sha256:" + hex.EncodeToString(digest[:]),
		Current: ref.Revision == nil,
	}, nil
}

func (*memoryArtifactRepository) ListMetadata(context.Context, artifacts.Scope, artifacts.BindingPageQuery) ([]artifacts.Metadata, error) {
	return nil, artifacts.ErrQueryUnsupported
}

func (*memoryArtifactRepository) ListVersions(context.Context, artifacts.Scope, artifacts.ArtifactRef, artifacts.VersionPageQuery) ([]artifacts.Metadata, error) {
	return nil, artifacts.ErrQueryUnsupported
}

func (*memoryArtifactRepository) ListLineage(context.Context, artifacts.Scope, artifacts.ArtifactRef, artifacts.LineagePageQuery) ([]artifacts.LineageEdge, error) {
	return nil, artifacts.ErrQueryUnsupported
}

func (r *memoryArtifactRepository) ForkSkill(
	_ context.Context,
	sourceScope artifacts.Scope,
	sourceRef artifacts.ArtifactRef,
	targetScope artifacts.Scope,
	name string,
) (artifacts.ForkResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if sourceRef.Revision == nil {
		return artifacts.ForkResult{}, artifacts.ErrExactRevisionRequired
	}
	source, ok := r.history[artifactKey(sourceScope, sourceRef)][*sourceRef.Revision]
	if !ok {
		return artifacts.ForkResult{}, artifacts.ErrArtifactNotFound
	}
	targetKey := artifactKey(targetScope, artifacts.ArtifactRef{Namespace: SkillNamespace, Name: name})
	if existing, exists := r.bindings[targetKey]; exists {
		previous := r.forkSources[targetKey]
		if previous.Revision == nil || *previous.Revision != *sourceRef.Revision {
			return artifacts.ForkResult{}, artifacts.ErrArtifactConflict
		}
		return artifacts.ForkResult{
			SourceRef: source.result.Ref, TargetRef: existing.result.Ref,
			MediaType: source.result.Payload.MediaType, Size: int64(len(source.result.Payload.Data)),
		}, nil
	}
	r.forkCalls++
	r.revisions++
	revision := fmt.Sprintf("rev_%d", r.revisions)
	exact := revision
	target := memoryArtifact{result: artifacts.ReadResult{
		Ref:     artifacts.ArtifactRef{Namespace: SkillNamespace, Name: name, Revision: &exact},
		Payload: source.result.Payload,
	}}
	r.bindings[targetKey] = target
	r.history[targetKey] = map[string]memoryArtifact{revision: target}
	r.forkSources[targetKey] = source.result.Ref
	return artifacts.ForkResult{
		SourceRef: source.result.Ref, TargetRef: target.result.Ref,
		MediaType: source.result.Payload.MediaType, Size: int64(len(source.result.Payload.Data)),
	}, nil
}
