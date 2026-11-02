package artifacts

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/apierror"
)

var (
	// ErrHistoryProtected reports that the post-publication history of a
	// binding cannot fit its limits even after every eligible revision is
	// removed: its current, the new revision and its protected revisions
	// alone exceed them. Nothing changed; only an authorized hold release
	// or a higher configured limit can make the same write succeed.
	ErrHistoryProtected = apierror.New(apierror.ArtifactHistoryProtected,
		"artifact history is protected and cannot fit its retention limit")
	// ErrHistoryMaintenancePending reports that admission needs more history
	// maintenance than one transaction may do. The write stored nothing; any
	// maintenance already committed for the binding stays committed, and
	// a later attempt continues it.
	ErrHistoryMaintenancePending = apierror.New(apierror.ArtifactHistoryMaintenancePending,
		"artifact history maintenance is pending")
)

// HistoryMaintenanceError is ErrHistoryMaintenancePending for one binding. A
// write in a transaction the store owns commits one maintenance pass before
// it returns the error. A write in a caller's transaction cannot: the caller
// rolls its transaction back and passes the error to
// PostgresRepository.MaintainHistory, which commits that pass.
type HistoryMaintenanceError struct {
	Scope Scope
	Ref   ArtifactRef
	// Size is the encoded size of the revision whose admission is pending.
	Size int64
	// Removed counts the revisions the committed maintenance pass removed.
	Removed int
	// Maintained reports whether that pass committed.
	Maintained bool
	// Cause is the failure of a maintenance pass that did not commit.
	Cause error
}

func (e *HistoryMaintenanceError) Error() string {
	switch {
	case e.Cause != nil:
		return fmt.Sprintf("artifact %s/%s history maintenance is pending; its pass failed: %v",
			e.Ref.Namespace, e.Ref.Name, e.Cause)
	case !e.Maintained:
		return fmt.Sprintf("artifact %s/%s history maintenance is pending", e.Ref.Namespace, e.Ref.Name)
	default:
		return fmt.Sprintf("artifact %s/%s history maintenance is pending after removing %d revisions",
			e.Ref.Namespace, e.Ref.Name, e.Removed)
	}
}

func (e *HistoryMaintenanceError) Unwrap() []error {
	if e.Cause != nil {
		return []error{ErrHistoryMaintenancePending, e.Cause}
	}
	return []error{ErrHistoryMaintenancePending}
}

// HistoryProtectedError is ErrHistoryProtected with the trusted accounting
// that caused it.
type HistoryProtectedError struct {
	Ref ArtifactRef
	// RetainedRevisions and RetainedBytes are what the binding would retain
	// after the write with every eligible revision removed.
	RetainedRevisions int64
	RetainedBytes     int64
	MaxRevisions      int64
	MaxBytes          int64
}

func (e *HistoryProtectedError) Error() string {
	return fmt.Sprintf("artifact %s/%s protected history (%d revisions, %d bytes) exceeds %d revisions or %d bytes",
		e.Ref.Namespace, e.Ref.Name, e.RetainedRevisions, e.RetainedBytes, e.MaxRevisions, e.MaxBytes)
}

func (e *HistoryProtectedError) Unwrap() error { return ErrHistoryProtected }
