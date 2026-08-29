package artifacts

import (
	"errors"
	"fmt"
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
)

type ConflictError struct {
	Ref              ArtifactRef
	ExpectedRevision *string
}

func (e *ConflictError) Error() string {
	return fmt.Sprintf("artifact binding %s/%s did not satisfy its revision precondition", e.Ref.Namespace, e.Ref.Name)
}

func (e *ConflictError) Unwrap() error { return ErrArtifactConflict }
