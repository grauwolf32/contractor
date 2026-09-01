package memory

import (
	"context"
	"errors"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

var testEpoch = time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC)

func TestNamespaceWriteReplaceAppendListSearchAndTags(t *testing.T) {
	store := newMemoryArtifactStore()
	namespace := mustNamespace(t, store)
	ctx := t.Context()

	first, err := namespace.WriteMemory(
		ctx, "repo_overview", "first", "overview", []string{"repository", "architecture"},
	)
	if err != nil {
		t.Fatal(err)
	}
	second, err := namespace.WriteMemory(
		ctx, "findings", "finding one", "security", []string{"security", "repository"},
	)
	if err != nil {
		t.Fatal(err)
	}
	replaced, err := namespace.WriteMemory(ctx, "repo_overview", "replacement", "", []string{"architecture"})
	if err != nil {
		t.Fatal(err)
	}
	appended, err := namespace.AppendMemory(ctx, "findings", "finding two")
	if err != nil {
		t.Fatal(err)
	}

	if first.Ordinal != 0 || second.Ordinal != 1 || replaced.Ordinal != first.Ordinal ||
		!replaced.CreatedAt.Equal(first.CreatedAt) || !replaced.UpdatedAt.After(first.UpdatedAt) ||
		appended.Content != "finding one\nfinding two" || appended.Description != "security" ||
		!reflect.DeepEqual(appended.Tags, []string{"repository", "security"}) ||
		!appended.CreatedAt.Equal(second.CreatedAt) {
		t.Fatalf("Memory mutation projections: first=%+v second=%+v replaced=%+v appended=%+v",
			first, second, replaced, appended)
	}
	read, err := namespace.ReadMemory(ctx, "findings")
	if err != nil || !reflect.DeepEqual(read, appended) {
		t.Fatalf("ReadMemory = (%+v, %v), want %+v", read, err, appended)
	}
	listed, err := namespace.ListMemories(ctx)
	if err != nil || len(listed) != 2 || listed[0].Name != "findings" || listed[1].Name != "repo_overview" {
		t.Fatalf("ListMemories = (%+v, %v)", listed, err)
	}
	searched, err := namespace.SearchMemory(ctx, []string{"architecture", "security"})
	if err != nil || len(searched) != 2 {
		t.Fatalf("SearchMemory = (%+v, %v)", searched, err)
	}
	tags, err := namespace.ListMemoryTags(ctx)
	if err != nil || !reflect.DeepEqual(tags, []string{"architecture", "repository", "security"}) {
		t.Fatalf("ListMemoryTags = (%v, %v)", tags, err)
	}
}

func TestNamespaceResponseLossUsesExactReplayAndAppendOccursOnce(t *testing.T) {
	for _, fault := range []string{"unknown_before", "unknown_after"} {
		t.Run(fault, func(t *testing.T) {
			store := newMemoryArtifactStore()
			store.faults = []string{fault}
			namespace := mustNamespace(t, store)
			note, err := namespace.WriteMemory(t.Context(), "response_loss", "body", "", nil)
			if err != nil || note.Content != "body" {
				t.Fatalf("WriteMemory = (%+v, %v)", note, err)
			}
			if len(store.attempts) != 2 || !reflect.DeepEqual(store.attempts[0], store.attempts[1]) ||
				store.semanticWrites != 1 {
				t.Fatalf("attempts=%+v semanticWrites=%d", store.attempts, store.semanticWrites)
			}
		})
	}

	store := newMemoryArtifactStore()
	namespace := mustNamespace(t, store)
	if _, err := namespace.WriteMemory(t.Context(), "append_once", "line one", "", nil); err != nil {
		t.Fatal(err)
	}
	store.faults = append(store.faults, "unknown_after")
	before := len(store.attempts)
	appended, err := namespace.AppendMemory(t.Context(), "append_once", "line two")
	if err != nil || appended.Content != "line one\nline two" || store.semanticWrites != 2 {
		t.Fatalf("AppendMemory = (%+v, %v), semanticWrites=%d", appended, err, store.semanticWrites)
	}
	if attempts := store.attempts[before:]; len(attempts) != 2 || !reflect.DeepEqual(attempts[0], attempts[1]) {
		t.Fatalf("append replay attempts = %+v", attempts)
	}
}

func TestNamespaceCASInterferenceReturnsChangedWithoutOverwrite(t *testing.T) {
	store := newMemoryArtifactStore()
	namespace := mustNamespace(t, store)
	if _, err := namespace.WriteMemory(t.Context(), "shared", "base", "", nil); err != nil {
		t.Fatal(err)
	}
	store.advancePayload = encodedTestNote(t, "shared", "external", 0)
	store.faults = append(store.faults, "advance_before")
	_, err := namespace.AppendMemory(t.Context(), "shared", "ours")
	assertToolError(t, err, CodeChanged, true)
	if got := store.content("analysis", "memory.shared"); got != "external" {
		t.Fatalf("stale append overwrote current content: %q", got)
	}

	store.advancePayload = encodedTestNote(t, "shared", "newer external", 0)
	store.faults = append(store.faults, "unknown_after_advance")
	_, err = namespace.AppendMemory(t.Context(), "shared", "lost append")
	assertToolError(t, err, CodeChanged, true)
	if got := store.content("analysis", "memory.shared"); got != "newer external" {
		t.Fatalf("ambiguous append overwrote newer content: %q", got)
	}
}

func TestNamespaceReconciliationUsesCurrentReadAuthorityError(t *testing.T) {
	store := newMemoryArtifactStore()
	store.readFaults = []error{nil, ErrAccessForbidden}
	store.faults = []string{"unknown_before", "unknown_before"}
	namespace := mustNamespace(t, store)

	_, err := namespace.WriteMemory(t.Context(), "shared", "body", "", nil)
	assertToolError(t, err, CodeForbidden, false)
	if store.semanticWrites != 0 {
		t.Fatalf("ambiguous forbidden create committed %d semantic writes", store.semanticWrites)
	}
}

func TestNamespaceRejectsPurposeReservedBinding(t *testing.T) {
	for _, namespace := range []string{"inputs", "outputs", "skills"} {
		_, err := NewNamespace(newMemoryArtifactStore(), Binding{
			RunID: "run-1", StageExecutionID: "stage-1", Namespace: namespace,
		})
		if err == nil {
			t.Fatalf("NewNamespace accepted purpose-reserved Namespace %q", namespace)
		}
	}
}

func TestNamespaceQuotaErrorsAndCancelledAuthorityAreBounded(t *testing.T) {
	store := newMemoryArtifactStore()
	for ordinal := 0; ordinal < MaximumNotes; ordinal++ {
		store.seed(t, "analysis", "note_"+decimal(ordinal), "body", uint64(ordinal))
	}
	namespace := mustNamespace(t, store)
	_, err := namespace.WriteMemory(t.Context(), "overflow", "body", "", nil)
	assertToolError(t, err, CodeNamespaceFull, false)
	updated, err := namespace.WriteMemory(t.Context(), "note_0", "updated", "", nil)
	if err != nil || updated.Ordinal != 0 {
		t.Fatalf("update at capacity = (%+v, %v)", updated, err)
	}

	_, err = namespace.ReadMemory(t.Context(), "missing")
	assertToolError(t, err, CodeNotFound, false)
	_, err = namespace.SearchMemory(t.Context(), []string{"same", "same"})
	assertToolError(t, err, CodeInvalid, false)
	store.forbidden = true
	_, err = namespace.ReadMemory(t.Context(), "note_0")
	assertToolError(t, err, CodeForbidden, false)
	store.forbidden = false
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = namespace.ListMemories(cancelled)
	assertToolError(t, err, CodeForbidden, false)

	corruptStore := newMemoryArtifactStore()
	corruptStore.seedRaw("analysis", "memory.corrupt", []byte(`{"not":"canonical"}`), MediaType)
	_, err = mustNamespace(t, corruptStore).ListMemories(t.Context())
	assertToolError(t, err, CodeUnavailable, true)
	if strings.Contains(err.Error(), "canonical") {
		t.Fatalf("bounded error leaked stored bytes: %q", err)
	}
}

func TestNamespaceSerializesConcurrentCalls(t *testing.T) {
	store := newMemoryArtifactStore()
	store.operationDelay = time.Millisecond
	namespace := mustNamespace(t, store)
	var wait sync.WaitGroup
	for _, name := range []string{"first", "second"} {
		wait.Add(1)
		go func() {
			defer wait.Done()
			if _, err := namespace.WriteMemory(t.Context(), name, "body", "", nil); err != nil {
				t.Errorf("WriteMemory(%q): %v", name, err)
			}
		}()
	}
	wait.Wait()
	if store.maximumActive != 1 {
		t.Fatalf("maximum concurrent Store operations = %d, want 1", store.maximumActive)
	}
	listed, err := namespace.ListMemories(t.Context())
	if err != nil || len(listed) != 2 || listed[0].Ordinal == listed[1].Ordinal {
		t.Fatalf("serialized notes = (%+v, %v)", listed, err)
	}
}

func TestNamespaceRejectsImpossibleOrdinalSetsOnlyForGlobalViewsAndCreates(t *testing.T) {
	for _, test := range []struct {
		name     string
		ordinals []uint64
	}{
		{name: "duplicate", ordinals: []uint64{0, 0}},
		{name: "gap", ordinals: []uint64{0, 2}},
	} {
		t.Run(test.name, func(t *testing.T) {
			store := newMemoryArtifactStore()
			store.seed(t, "analysis", "first", "first body", test.ordinals[0])
			store.seed(t, "analysis", "second", "second body", test.ordinals[1])
			namespace := mustNamespace(t, store)

			read, err := namespace.ReadMemory(t.Context(), "first")
			if err != nil || read.Content != "first body" {
				t.Fatalf("targeted read = (%+v, %v)", read, err)
			}
			updated, err := namespace.WriteMemory(t.Context(), "first", "updated", "", nil)
			if err != nil || updated.Content != "updated" || updated.Ordinal != test.ordinals[0] {
				t.Fatalf("targeted update = (%+v, %v)", updated, err)
			}
			for operation, invoke := range map[string]func() error{
				"list": func() error {
					_, err := namespace.ListMemories(t.Context())
					return err
				},
				"search": func() error {
					_, err := namespace.SearchMemory(t.Context(), []string{"tag"})
					return err
				},
				"tags": func() error {
					_, err := namespace.ListMemoryTags(t.Context())
					return err
				},
				"create": func() error {
					_, err := namespace.WriteMemory(t.Context(), "third", "body", "", nil)
					return err
				},
			} {
				t.Run(operation, func(t *testing.T) {
					assertToolError(t, invoke(), CodeUnavailable, true)
				})
			}
		})
	}
}

func TestNamespaceAppendUsesOnlyTheRealFinalPayloadBound(t *testing.T) {
	shortName := "a"
	existing := "x"
	fragment := maximumFittingAppendFragment(t, shortName, existing)
	if len(fragment) < MaximumPayloadBytes/2 {
		t.Fatalf("boundary fragment is unexpectedly short: %d", len(fragment))
	}
	shortStore := newMemoryArtifactStore()
	shortNamespace := mustNamespace(t, shortStore)
	if _, err := shortNamespace.WriteMemory(t.Context(), shortName, existing, "", nil); err != nil {
		t.Fatal(err)
	}
	appended, err := shortNamespace.AppendMemory(t.Context(), shortName, fragment)
	if err != nil || appended.Content != existing+"\n"+fragment {
		t.Fatalf("short-name boundary append = (bytes=%d, %v)", len(fragment), err)
	}

	longName := "a" + strings.Repeat("b", MaximumNameBytes-1)
	longStore := newMemoryArtifactStore()
	longNamespace := mustNamespace(t, longStore)
	if _, err := longNamespace.WriteMemory(t.Context(), longName, existing, "", nil); err != nil {
		t.Fatal(err)
	}
	_, err = longNamespace.AppendMemory(t.Context(), longName, fragment)
	assertToolError(t, err, CodeTooLarge, false)
	for _, invalid := range []string{"", string([]byte{0xff})} {
		_, err = shortNamespace.AppendMemory(t.Context(), shortName, invalid)
		assertToolError(t, err, CodeInvalid, false)
	}
}

func TestOrderedNotesUsesUpdatedOrdinalAndNameTotalOrder(t *testing.T) {
	timestamp := testEpoch
	notes := []loadedNote{
		{note: StoredNote{Name: "zeta", Ordinal: 1}, revisionCreatedAt: timestamp},
		{note: StoredNote{Name: "beta", Ordinal: 2}, revisionCreatedAt: timestamp},
		{note: StoredNote{Name: "alpha", Ordinal: 2}, revisionCreatedAt: timestamp},
		{note: StoredNote{Name: "newest", Ordinal: 0}, revisionCreatedAt: timestamp.Add(time.Second)},
	}
	orderedNotes(notes)
	got := make([]string, len(notes))
	for index := range notes {
		got[index] = notes[index].note.Name
	}
	if want := []string{"newest", "alpha", "beta", "zeta"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("total order = %v, want %v", got, want)
	}
}

func TestMemoryToolErrorsNormalizeToClosedRetryabilityTable(t *testing.T) {
	for _, test := range []struct {
		inputCode      string
		inputRetryable bool
		wantCode       string
		wantRetryable  bool
	}{
		{CodeInvalid, true, CodeInvalid, false},
		{CodeChanged, false, CodeChanged, true},
		{CodeForbidden, true, CodeForbidden, false},
		{"internal_arbitrary", false, CodeUnavailable, true},
	} {
		bounded := mapStoreError(&ToolError{Code: test.inputCode, Retryable: test.inputRetryable})
		assertToolError(t, bounded, test.wantCode, test.wantRetryable)
		direct := NormalizeToolError(&ToolError{
			Code: test.inputCode, Retryable: test.inputRetryable,
		})
		assertToolError(t, direct, test.wantCode, test.wantRetryable)
	}
	assertToolError(t, NormalizeToolError(errors.New("arbitrary internal failure")), CodeUnavailable, true)
}

func maximumFittingAppendFragment(t *testing.T, name, existing string) string {
	t.Helper()
	low, high := 1, MaximumPayloadBytes
	for low < high {
		middle := low + (high-low+1)/2
		_, err := Encode(StoredNote{
			SchemaVersion: SchemaVersion, Name: name,
			Content: existing + "\n" + strings.Repeat("x", middle), Tags: []string{},
		})
		if err == nil {
			low = middle
		} else {
			high = middle - 1
		}
	}
	return strings.Repeat("x", low)
}

func mustNamespace(t *testing.T, store Store) *Namespace {
	t.Helper()
	result, err := NewNamespace(store, Binding{
		RunID: "run-1", StageExecutionID: "stage-1", Namespace: "analysis",
	})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func assertToolError(t *testing.T, err error, code string, retryable bool) {
	t.Helper()
	var bounded *ToolError
	if !errors.As(err, &bounded) || bounded.Code != code || bounded.Retryable != retryable {
		t.Fatalf("error = %#v, want %s retryable=%t", err, code, retryable)
	}
}

type memoryWriteAttempt struct {
	ref              artifacts.ArtifactRef
	payload          artifacts.Payload
	expectedRevision *string
}

type storedMemoryArtifact struct {
	revision          string
	payload           artifacts.Payload
	bindingCreatedAt  time.Time
	revisionCreatedAt time.Time
}

type memoryArtifactStore struct {
	mu             sync.Mutex
	activityMu     sync.Mutex
	bindings       map[string]storedMemoryArtifact
	nextRevision   int
	clock          int
	faults         []string
	advancePayload []byte
	attempts       []memoryWriteAttempt
	semanticWrites int
	forbidden      bool
	active         int
	maximumActive  int
	operationDelay time.Duration
	readFaults     []error
}

func newMemoryArtifactStore() *memoryArtifactStore {
	return &memoryArtifactStore{bindings: map[string]storedMemoryArtifact{}}
}

func (s *memoryArtifactStore) List(
	_ context.Context, binding Binding,
) ([]artifacts.ArtifactRef, error) {
	s.begin()
	defer s.end()
	if s.forbidden {
		return nil, ErrAccessForbidden
	}
	result := make([]artifacts.ArtifactRef, 0, len(s.bindings))
	for key := range s.bindings {
		namespace, name, _ := strings.Cut(key, "/")
		if namespace == binding.Namespace {
			result = append(result, artifacts.ArtifactRef{Namespace: namespace, Name: name})
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result, nil
}

func (s *memoryArtifactStore) Read(
	_ context.Context, binding Binding, ref artifacts.ArtifactRef,
) (artifacts.ReadResult, error) {
	s.begin()
	defer s.end()
	if len(s.readFaults) > 0 {
		fault := s.readFaults[0]
		s.readFaults = s.readFaults[1:]
		if fault != nil {
			return artifacts.ReadResult{}, fault
		}
	}
	if s.forbidden {
		return artifacts.ReadResult{}, ErrAccessForbidden
	}
	stored, ok := s.bindings[binding.Namespace+"/"+ref.Name]
	if !ok {
		return artifacts.ReadResult{}, artifacts.ErrArtifactNotFound
	}
	revision := stored.revision
	return artifacts.ReadResult{
		Ref: artifacts.ArtifactRef{Namespace: binding.Namespace, Name: ref.Name, Revision: &revision},
		Payload: artifacts.Payload{
			MediaType: stored.payload.MediaType, Data: append([]byte(nil), stored.payload.Data...),
		},
		BindingCreatedAt: stored.bindingCreatedAt, RevisionCreatedAt: stored.revisionCreatedAt,
	}, nil
}

func (s *memoryArtifactStore) Write(
	_ context.Context,
	binding Binding,
	ref artifacts.ArtifactRef,
	payload artifacts.Payload,
	expectedRevision *string,
) (artifacts.WriteResult, error) {
	s.begin()
	defer s.end()
	if s.forbidden {
		return artifacts.WriteResult{}, ErrAccessForbidden
	}
	attempt := memoryWriteAttempt{
		ref: ref,
		payload: artifacts.Payload{
			MediaType: payload.MediaType, Data: append([]byte(nil), payload.Data...),
		},
		expectedRevision: cloneString(expectedRevision),
	}
	s.attempts = append(s.attempts, attempt)
	fault := ""
	if len(s.faults) > 0 {
		fault, s.faults = s.faults[0], s.faults[1:]
	}
	if fault == "unknown_before" {
		return artifacts.WriteResult{}, ErrMutationOutcomeUnknown
	}
	if fault == "advance_before" {
		s.advance(binding.Namespace, ref.Name)
	}
	stored, err := s.apply(binding.Namespace, attempt)
	if err != nil {
		return artifacts.WriteResult{}, err
	}
	if fault == "unknown_after_advance" {
		s.advance(binding.Namespace, ref.Name)
	}
	if fault == "unknown_after" || fault == "unknown_after_advance" {
		return artifacts.WriteResult{}, ErrMutationOutcomeUnknown
	}
	revision := stored.revision
	return artifacts.WriteResult{
		Ref:       artifacts.ArtifactRef{Namespace: binding.Namespace, Name: ref.Name, Revision: &revision},
		MediaType: stored.payload.MediaType, Size: int64(len(stored.payload.Data)),
		BindingCreatedAt: stored.bindingCreatedAt, RevisionCreatedAt: stored.revisionCreatedAt,
	}, nil
}

func (s *memoryArtifactStore) apply(
	namespace string, attempt memoryWriteAttempt,
) (storedMemoryArtifact, error) {
	key := namespace + "/" + attempt.ref.Name
	current, exists := s.bindings[key]
	if attempt.expectedRevision == nil && exists ||
		attempt.expectedRevision != nil && (!exists || *attempt.expectedRevision != current.revision) {
		return storedMemoryArtifact{}, artifacts.ErrArtifactConflict
	}
	createdAt := current.bindingCreatedAt
	if !exists {
		createdAt = s.tick()
	}
	s.nextRevision++
	stored := storedMemoryArtifact{
		revision: "revision-" + decimal(s.nextRevision),
		payload: artifacts.Payload{
			MediaType: attempt.payload.MediaType, Data: append([]byte(nil), attempt.payload.Data...),
		},
		bindingCreatedAt: createdAt, revisionCreatedAt: s.tick(),
	}
	s.bindings[key] = stored
	s.semanticWrites++
	return stored, nil
}

func (s *memoryArtifactStore) advance(namespace, name string) {
	key := namespace + "/" + name
	current, ok := s.bindings[key]
	if !ok || s.advancePayload == nil {
		panic("advance fault requires current binding and payload")
	}
	s.nextRevision++
	s.bindings[key] = storedMemoryArtifact{
		revision: "revision-" + decimal(s.nextRevision),
		payload: artifacts.Payload{
			MediaType: MediaType, Data: append([]byte(nil), s.advancePayload...),
		},
		bindingCreatedAt: current.bindingCreatedAt, revisionCreatedAt: s.tick(),
	}
	s.semanticWrites++
}

func (s *memoryArtifactStore) seed(
	t *testing.T, namespace, name, content string, ordinal uint64,
) {
	t.Helper()
	s.seedRaw(namespace, ArtifactNamePrefix+name, encodedTestNote(t, name, content, ordinal), MediaType)
}

func (s *memoryArtifactStore) seedRaw(namespace, name string, data []byte, mediaType string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextRevision++
	timestamp := s.tick()
	s.bindings[namespace+"/"+name] = storedMemoryArtifact{
		revision:         "revision-" + decimal(s.nextRevision),
		payload:          artifacts.Payload{MediaType: mediaType, Data: append([]byte(nil), data...)},
		bindingCreatedAt: timestamp, revisionCreatedAt: timestamp,
	}
}

func (s *memoryArtifactStore) content(namespace, name string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	note, err := Decode(name, s.bindings[namespace+"/"+name].payload.Data)
	if err != nil {
		panic(err)
	}
	return note.Content
}

func (s *memoryArtifactStore) begin() {
	s.activityMu.Lock()
	s.active++
	s.maximumActive = max(s.maximumActive, s.active)
	s.activityMu.Unlock()
	if s.operationDelay > 0 {
		time.Sleep(s.operationDelay)
	}
	s.mu.Lock()
}

func (s *memoryArtifactStore) end() {
	s.mu.Unlock()
	s.activityMu.Lock()
	s.active--
	s.activityMu.Unlock()
}

func (s *memoryArtifactStore) tick() time.Time {
	s.clock++
	return testEpoch.Add(time.Duration(s.clock) * time.Microsecond)
}

func encodedTestNote(t *testing.T, name, content string, ordinal uint64) []byte {
	t.Helper()
	payload, err := Encode(StoredNote{
		SchemaVersion: SchemaVersion, Name: name, Content: content, Tags: []string{}, Ordinal: ordinal,
	})
	if err != nil {
		t.Fatal(err)
	}
	return payload
}

func decimal(value int) string {
	if value == 0 {
		return "0"
	}
	var digits [20]byte
	index := len(digits)
	for value > 0 {
		index--
		digits[index] = byte('0' + value%10)
		value /= 10
	}
	return string(digits[index:])
}

var _ Store = (*memoryArtifactStore)(nil)
