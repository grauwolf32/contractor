package memory

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

const (
	ToolsetRef = "memory-tools@1"

	ToolListMemories   = "list_memories"
	ToolReadMemory     = "read_memory"
	ToolWriteMemory    = "write_memory"
	ToolAppendMemory   = "append_memory"
	ToolSearchMemory   = "search_memory"
	ToolListMemoryTags = "list_memory_tags"

	MaximumNotes = 128
)

var operationNames = [...]string{
	ToolListMemories,
	ToolReadMemory,
	ToolWriteMemory,
	ToolAppendMemory,
	ToolSearchMemory,
	ToolListMemoryTags,
}

const (
	CodeInvalid       = "memory_invalid"
	CodeNotFound      = "memory_not_found"
	CodeTooLarge      = "memory_too_large"
	CodeNamespaceFull = "memory_namespace_full"
	CodeChanged       = "memory_changed"
	CodeForbidden     = "memory_forbidden"
	CodeUnavailable   = "memory_unavailable"
)

var (
	// ErrAccessForbidden is returned by a Store when its Stage-bound authority
	// no longer permits the operation. It deliberately carries no Stage state.
	ErrAccessForbidden = errors.New("memory access is forbidden")
	// ErrMutationOutcomeUnknown means the caller did not observe whether the
	// exact Artifact mutation committed. Namespace performs the one permitted
	// byte-identical replay and reconciliation.
	ErrMutationOutcomeUnknown = errors.New("memory mutation outcome is unknown")
)

// ToolError is the complete bounded model-facing Memory error vocabulary.
type ToolError struct {
	Code      string
	Retryable bool
}

func (e *ToolError) Error() string { return "Memory operation failed (" + e.Code + ")" }

func OperationNames() []string { return append([]string(nil), operationNames[:]...) }

func IsOperation(name string) bool {
	for _, candidate := range operationNames {
		if name == candidate {
			return true
		}
	}
	return false
}

// Binding is trusted adapter state, never a model-visible tool argument.
type Binding struct {
	RunID            string
	StageExecutionID string
	Namespace        string
}

// Store is the Artifact persistence boundary used by a Stage-bound Namespace.
// A production Write must check Stage authority and mutate the binding in one
// transaction.
type Store interface {
	List(context.Context, Binding) ([]artifacts.ArtifactRef, error)
	Read(context.Context, Binding, artifacts.ArtifactRef) (artifacts.ReadResult, error)
	Write(
		context.Context,
		Binding,
		artifacts.ArtifactRef,
		artifacts.Payload,
		*string,
	) (artifacts.WriteResult, error)
}

// Namespace implements the six logical Memory operations over ordinary
// immutable-versioned Run artifacts. All operations on one bound view are
// serialized even though the current Planner invokes only one function at a
// time.
type Namespace struct {
	store   Store
	binding Binding
	mu      sync.Mutex
}

func NewNamespace(store Store, binding Binding) (*Namespace, error) {
	if store == nil || !validBinding(binding) {
		return nil, fmt.Errorf("Memory Store and valid bound Run, StageExecution and Namespace are required")
	}
	return &Namespace{store: store, binding: binding}, nil
}

func (n *Namespace) ListMemories(ctx context.Context) ([]Preview, error) {
	if err := activeContext(ctx); err != nil {
		return nil, err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	notes, err := n.loadAll(ctx)
	if err != nil {
		return nil, err
	}
	orderedNotes(notes)
	result := make([]Preview, len(notes))
	for index := range notes {
		result[index] = PreviewProjection(
			notes[index].note, notes[index].bindingCreatedAt, notes[index].revisionCreatedAt,
		)
	}
	return result, nil
}

func (n *Namespace) ReadMemory(ctx context.Context, name string) (Note, error) {
	if err := activeContext(ctx); err != nil {
		return Note{}, err
	}
	bindingName, err := validatedArtifactName(name)
	if err != nil {
		return Note{}, err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	loaded, err := n.readOptional(ctx, bindingName)
	if err != nil {
		return Note{}, err
	}
	if loaded == nil {
		return Note{}, toolError(CodeNotFound, false)
	}
	return FullProjection(loaded.note, loaded.bindingCreatedAt, loaded.revisionCreatedAt), nil
}

func (n *Namespace) WriteMemory(
	ctx context.Context,
	name string,
	content string,
	description string,
	tags []string,
) (Note, error) {
	if err := activeContext(ctx); err != nil {
		return Note{}, err
	}
	bindingName, err := validatedArtifactName(name)
	if err != nil {
		return Note{}, err
	}
	if _, _, err := encodeInput(name, content, description, tags, 0); err != nil {
		return Note{}, err
	}
	n.mu.Lock()
	defer n.mu.Unlock()

	existing, err := n.readOptional(ctx, bindingName)
	if err != nil {
		return Note{}, err
	}
	var ordinal uint64
	var expectedRevision *string
	if existing == nil {
		current, err := n.loadAll(ctx)
		if err != nil {
			return Note{}, err
		}
		for index := range current {
			if current[index].note.Name == name {
				existing = &current[index]
				break
			}
		}
		if existing == nil {
			if len(current) >= MaximumNotes {
				return Note{}, toolError(CodeNamespaceFull, false)
			}
			var found bool
			for _, item := range current {
				if !found || item.note.Ordinal >= ordinal {
					ordinal = item.note.Ordinal + 1
					found = true
				}
			}
			if ordinal > MaximumExactOrdinal {
				return Note{}, toolError(CodeNamespaceFull, false)
			}
		} else {
			ordinal = existing.note.Ordinal
			expectedRevision = stringPointer(existing.revision)
		}
	} else {
		ordinal = existing.note.Ordinal
		expectedRevision = stringPointer(existing.revision)
	}
	note, payload, err := encodeInput(name, content, description, tags, ordinal)
	if err != nil {
		return Note{}, err
	}
	metadata, err := n.writeExact(ctx, bindingName, payload, expectedRevision)
	if err != nil {
		return Note{}, err
	}
	return FullProjection(note, metadata.bindingCreatedAt, metadata.revisionCreatedAt), nil
}

func (n *Namespace) AppendMemory(ctx context.Context, name, content string) (Note, error) {
	if err := activeContext(ctx); err != nil {
		return Note{}, err
	}
	bindingName, err := validatedArtifactName(name)
	if err != nil {
		return Note{}, err
	}
	if _, _, err := encodeInput("append_validation", content, "", nil, 0); err != nil {
		return Note{}, err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	existing, err := n.readOptional(ctx, bindingName)
	if err != nil {
		return Note{}, err
	}
	if existing == nil {
		return Note{}, toolError(CodeNotFound, false)
	}
	note, payload, err := encodeInput(
		existing.note.Name,
		existing.note.Content+"\n"+content,
		existing.note.Description,
		existing.note.Tags,
		existing.note.Ordinal,
	)
	if err != nil {
		return Note{}, err
	}
	metadata, err := n.writeExact(ctx, bindingName, payload, stringPointer(existing.revision))
	if err != nil {
		return Note{}, err
	}
	return FullProjection(note, metadata.bindingCreatedAt, metadata.revisionCreatedAt), nil
}

func (n *Namespace) SearchMemory(ctx context.Context, tags []string) ([]Preview, error) {
	if err := activeContext(ctx); err != nil {
		return nil, err
	}
	selected, err := validatedSearchTags(tags)
	if err != nil {
		return nil, err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	notes, err := n.loadAll(ctx)
	if err != nil {
		return nil, err
	}
	matched := make([]loadedNote, 0, len(notes))
	for _, item := range notes {
		if hasAnyTag(item.note.Tags, selected) {
			matched = append(matched, item)
		}
	}
	orderedNotes(matched)
	result := make([]Preview, len(matched))
	for index := range matched {
		result[index] = PreviewProjection(
			matched[index].note, matched[index].bindingCreatedAt, matched[index].revisionCreatedAt,
		)
	}
	return result, nil
}

func (n *Namespace) ListMemoryTags(ctx context.Context) ([]string, error) {
	if err := activeContext(ctx); err != nil {
		return nil, err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	notes, err := n.loadAll(ctx)
	if err != nil {
		return nil, err
	}
	unique := make(map[string]struct{})
	for _, item := range notes {
		for _, tag := range item.note.Tags {
			unique[tag] = struct{}{}
		}
	}
	result := make([]string, 0, len(unique))
	for tag := range unique {
		result = append(result, tag)
	}
	sort.Strings(result)
	return result, nil
}

type loadedNote struct {
	note              StoredNote
	bindingCreatedAt  time.Time
	revisionCreatedAt time.Time
	revision          string
	payload           []byte
}

type writeMetadata struct {
	bindingCreatedAt  time.Time
	revisionCreatedAt time.Time
}

func (n *Namespace) loadAll(ctx context.Context) ([]loadedNote, error) {
	refs, err := n.store.List(ctx, n.binding)
	if err != nil {
		return nil, mapStoreError(err)
	}
	memoryRefs := make([]artifacts.ArtifactRef, 0, len(refs))
	for _, ref := range refs {
		if strings.HasPrefix(ref.Name, ArtifactNamePrefix) {
			memoryRefs = append(memoryRefs, ref)
		}
	}
	sort.Slice(memoryRefs, func(i, j int) bool {
		if memoryRefs[i].Namespace == memoryRefs[j].Namespace {
			return memoryRefs[i].Name < memoryRefs[j].Name
		}
		return memoryRefs[i].Namespace < memoryRefs[j].Namespace
	})
	if len(memoryRefs) > MaximumNotes {
		return nil, toolError(CodeUnavailable, true)
	}
	result := make([]loadedNote, 0, len(memoryRefs))
	seen := make(map[string]struct{}, len(memoryRefs))
	for _, ref := range memoryRefs {
		if ref.Namespace != n.binding.Namespace || ref.Revision != nil {
			return nil, toolError(CodeUnavailable, true)
		}
		if _, duplicate := seen[ref.Name]; duplicate {
			return nil, toolError(CodeUnavailable, true)
		}
		seen[ref.Name] = struct{}{}
		loaded, err := n.readOptional(ctx, ref.Name)
		if err != nil {
			return nil, err
		}
		if loaded == nil {
			return nil, toolError(CodeUnavailable, true)
		}
		result = append(result, *loaded)
	}
	return result, nil
}

func (n *Namespace) readOptional(ctx context.Context, bindingName string) (*loadedNote, error) {
	result, err := n.store.Read(ctx, n.binding, artifacts.ArtifactRef{
		Namespace: n.binding.Namespace,
		Name:      bindingName,
	})
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, mapStoreError(err)
	}
	return decodeRead(n.binding.Namespace, bindingName, result)
}

func decodeRead(namespace, bindingName string, value artifacts.ReadResult) (*loadedNote, error) {
	if value.Ref.Namespace != namespace || value.Ref.Name != bindingName || value.Ref.Revision == nil ||
		value.Payload.MediaType != MediaType || value.BindingCreatedAt.IsZero() ||
		value.RevisionCreatedAt.IsZero() || value.RevisionCreatedAt.Before(value.BindingCreatedAt) {
		return nil, toolError(CodeUnavailable, true)
	}
	note, err := Decode(bindingName, value.Payload.Data)
	if err != nil {
		return nil, toolError(CodeUnavailable, true)
	}
	return &loadedNote{
		note: note, bindingCreatedAt: value.BindingCreatedAt,
		revisionCreatedAt: value.RevisionCreatedAt, revision: *value.Ref.Revision,
		payload: append([]byte(nil), value.Payload.Data...),
	}, nil
}

func (n *Namespace) writeExact(
	ctx context.Context,
	bindingName string,
	payload []byte,
	expectedRevision *string,
) (writeMetadata, error) {
	target := artifacts.ArtifactRef{Namespace: n.binding.Namespace, Name: bindingName}
	write := func() (artifacts.WriteResult, error) {
		return n.store.Write(
			ctx, n.binding, target,
			artifacts.Payload{MediaType: MediaType, Data: append([]byte(nil), payload...)},
			cloneString(expectedRevision),
		)
	}
	result, err := write()
	if err == nil {
		return validateWrite(target, payload, result)
	}
	if !errors.Is(err, ErrMutationOutcomeUnknown) {
		return writeMetadata{}, mapStoreError(err)
	}

	// Only an unknown commit outcome authorizes one exact replay. The immutable
	// bytes and create/update precondition are deliberately not recomputed.
	retryResult, retryErr := write()
	if retryErr == nil {
		return validateWrite(target, payload, retryResult)
	}
	current, readErr := n.readOptional(ctx, bindingName)
	if readErr == nil && current != nil && bytes.Equal(current.payload, payload) {
		return writeMetadata{
			bindingCreatedAt: current.bindingCreatedAt, revisionCreatedAt: current.revisionCreatedAt,
		}, nil
	}
	if readErr != nil {
		return writeMetadata{}, mapStoreError(retryErr)
	}
	return writeMetadata{}, toolError(CodeChanged, true)
}

func validateWrite(
	target artifacts.ArtifactRef,
	payload []byte,
	result artifacts.WriteResult,
) (writeMetadata, error) {
	if result.Ref.Namespace != target.Namespace || result.Ref.Name != target.Name ||
		result.Ref.Revision == nil || result.MediaType != MediaType || result.Size != int64(len(payload)) ||
		result.BindingCreatedAt.IsZero() || result.RevisionCreatedAt.IsZero() ||
		result.RevisionCreatedAt.Before(result.BindingCreatedAt) {
		return writeMetadata{}, toolError(CodeUnavailable, true)
	}
	return writeMetadata{
		bindingCreatedAt: result.BindingCreatedAt, revisionCreatedAt: result.RevisionCreatedAt,
	}, nil
}

func encodeInput(
	name, content, description string,
	tags []string,
	ordinal uint64,
) (StoredNote, []byte, error) {
	note, err := Normalize(StoredNote{
		SchemaVersion: SchemaVersion, Name: name, Content: content,
		Description: description, Tags: append([]string(nil), tags...), Ordinal: ordinal,
	})
	if err != nil {
		return StoredNote{}, nil, inputError(err)
	}
	payload, err := Encode(note)
	if err != nil {
		return StoredNote{}, nil, inputError(err)
	}
	return note, payload, nil
}

func inputError(err error) error {
	reason, _ := ErrorReasonOf(err)
	if reason == ReasonPayloadTooLarge {
		return toolError(CodeTooLarge, false)
	}
	return toolError(CodeInvalid, false)
}

func validatedArtifactName(name string) (string, error) {
	value, err := ArtifactName(name)
	if err != nil {
		return "", toolError(CodeInvalid, false)
	}
	return value, nil
}

func validatedSearchTags(tags []string) (map[string]struct{}, error) {
	if len(tags) == 0 {
		return nil, toolError(CodeInvalid, false)
	}
	note, err := Normalize(StoredNote{
		SchemaVersion: SchemaVersion, Name: "search_validation", Content: "x", Tags: tags,
	})
	if err != nil || len(note.Tags) != len(tags) {
		return nil, toolError(CodeInvalid, false)
	}
	result := make(map[string]struct{}, len(note.Tags))
	for _, tag := range note.Tags {
		result[tag] = struct{}{}
	}
	return result, nil
}

func orderedNotes(notes []loadedNote) {
	sort.Slice(notes, func(i, j int) bool {
		if !notes[i].revisionCreatedAt.Equal(notes[j].revisionCreatedAt) {
			return notes[i].revisionCreatedAt.After(notes[j].revisionCreatedAt)
		}
		if notes[i].note.Ordinal != notes[j].note.Ordinal {
			return notes[i].note.Ordinal > notes[j].note.Ordinal
		}
		return notes[i].note.Name < notes[j].note.Name
	})
}

func hasAnyTag(tags []string, selected map[string]struct{}) bool {
	for _, tag := range tags {
		if _, ok := selected[tag]; ok {
			return true
		}
	}
	return false
}

func activeContext(ctx context.Context) error {
	if ctx == nil || ctx.Err() != nil {
		return toolError(CodeForbidden, false)
	}
	return nil
}

func mapStoreError(err error) error {
	var bounded *ToolError
	if errors.As(err, &bounded) {
		return bounded
	}
	switch {
	case errors.Is(err, ErrAccessForbidden), errors.Is(err, artifacts.ErrArtifactFrozen),
		errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return toolError(CodeForbidden, false)
	case errors.Is(err, artifacts.ErrArtifactConflict):
		return toolError(CodeChanged, true)
	default:
		return toolError(CodeUnavailable, true)
	}
}

func toolError(code string, retryable bool) *ToolError {
	return &ToolError{Code: code, Retryable: retryable}
}

func validBinding(binding Binding) bool {
	return strings.TrimSpace(binding.RunID) != "" && !strings.ContainsRune(binding.RunID, 0) &&
		strings.TrimSpace(binding.StageExecutionID) != "" &&
		!strings.ContainsRune(binding.StageExecutionID, 0) &&
		strings.TrimSpace(binding.Namespace) != "" && !strings.Contains(binding.Namespace, "/") &&
		!strings.ContainsRune(binding.Namespace, 0)
}

func stringPointer(value string) *string { return &value }

func cloneString(value *string) *string {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}
