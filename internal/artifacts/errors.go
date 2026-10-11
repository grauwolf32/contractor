package artifacts

import (
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/clone"
)

var (
	ErrInvalidScope          = errors.New("invalid artifact scope")
	ErrInvalidName           = errors.New("invalid artifact namespace or name")
	ErrInvalidMediaType      = errors.New("invalid artifact media type")
	ErrPayloadTooLarge       = errors.New("artifact payload exceeds size limit")
	ErrVersionedWriteTarget  = errors.New("artifact write target must be versionless")
	ErrExactRevisionRequired = errors.New("exact artifact revision is required")
	ErrArtifactConflict      = errors.New("artifact binding compare-and-swap conflict")
	ErrArtifactNotFound      = errors.New("artifact not found")
	ErrReservedNamespace     = errors.New("artifact namespace is reserved")
	ErrArtifactFrozen        = errors.New("artifact binding is frozen")
	ErrArtifactIntegrity     = errors.New("artifact payload integrity failure")
	ErrQueryUnsupported      = errors.New("artifact query repository is unavailable")
	ErrScopeDeleting         = errors.New("artifact Project scope is deleting")
)

type ConflictError struct {
	Ref              ArtifactRef
	ExpectedRevision *string
}

func (e *ConflictError) Error() string {
	return fmt.Sprintf("artifact binding %s/%s did not satisfy its revision precondition", e.Ref.Namespace, e.Ref.Name)
}

func (e *ConflictError) Unwrap() error { return ErrArtifactConflict }

// WritePrecondition is the compare-and-set every binding write applies to the
// binding's current head, nil when the binding does not exist: a write that
// expects no revision only creates, and one that expects a revision advances
// exactly that current revision (ART-R58, ART-R59). A frozen binding refuses
// every write with ErrArtifactFrozen, whatever revision it expects (ART-R94,
// ART-R95); any other mismatch is a *ConflictError. It returns nil when the
// write may apply. Test doubles of the binding store apply it too, so they
// reject what the store rejects.
func WritePrecondition(target ArtifactRef, current *Metadata, expectedRevision *string) error {
	conflict := &ConflictError{Ref: target, ExpectedRevision: clone.Pointer(expectedRevision)}
	if current == nil {
		if expectedRevision != nil {
			return conflict
		}
		return nil
	}
	if current.Frozen {
		return ErrArtifactFrozen
	}
	if expectedRevision == nil || current.Ref.Revision == nil || *current.Ref.Revision != *expectedRevision {
		return conflict
	}
	return nil
}
