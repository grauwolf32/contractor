package artifacts

import (
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/clone"
)

// The public API answers these errors with the Kind each one reports
// (apierror); the private Artifact API keeps its own codes for the Runtime.
var (
	ErrInvalidScope          = apierror.New(apierror.InvalidRequest, "invalid artifact scope")
	ErrInvalidName           = apierror.New(apierror.InvalidRequest, "invalid artifact namespace or name")
	ErrInvalidMediaType      = apierror.New(apierror.InvalidRequest, "invalid artifact media type")
	ErrPayloadTooLarge       = apierror.New(apierror.ArtifactTooLarge, "artifact payload exceeds size limit")
	ErrVersionedWriteTarget  = apierror.New(apierror.InvalidRequest, "artifact write target must be versionless")
	ErrExactRevisionRequired = apierror.New(apierror.InvalidRequest, "exact artifact revision is required")
	ErrArtifactConflict      = apierror.New(apierror.Conflict, "artifact binding compare-and-swap conflict")
	ErrArtifactNotFound      = apierror.New(apierror.NotFound, "artifact not found")
	ErrReservedNamespace     = apierror.New(apierror.InvalidRequest, "artifact namespace is reserved")
	ErrArtifactFrozen        = apierror.New(apierror.ArtifactFrozen, "artifact binding is frozen")
	ErrArtifactIntegrity     = apierror.New(apierror.ArtifactContentCorrupt, "artifact payload integrity failure")
	ErrQueryUnsupported      = errors.New("artifact query repository is unavailable")
	ErrScopeDeleting         = apierror.New(apierror.ProjectDeleting, "artifact Project scope is deleting")
	// ErrRunRevisionLimit reports that a RunScope already holds the most
	// binding revisions one Run may create (LIM-ART-6). It is permanent for
	// that Run.
	ErrRunRevisionLimit = errors.New("Run artifact revision limit is exhausted")
)

// ErrAuditArtifactCollision means a content-named Audit artifact binding
// already holds other content. It is an integrity failure: retrying the same
// write can never succeed.
var ErrAuditArtifactCollision = fmt.Errorf("%w: Audit artifact binding holds other content", ErrArtifactIntegrity)

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
