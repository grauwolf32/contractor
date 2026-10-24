package artifacts

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
)

// The retained-history policy of every logical binding (ART-R122 to
// ART-R127): one typed definition shared by Server configuration, the
// ArtifactStore and its tests. The limits apply to what one binding (scope
// kind, scope ID, namespace, name) retains now, never to what it accepted
// over time, and to every binding of every scope without registration.
const (
	// DefaultHistoryMaxRevisions and DefaultHistoryMaxBytes are the retained
	// revisions and full encoded bytes of one binding (LIM-ART-9), counting
	// its current and protected revisions.
	DefaultHistoryMaxRevisions int64 = 2000
	DefaultHistoryMaxBytes     int64 = 1 << 30
	// HistoryRevisionsCeiling and HistoryBytesCeiling are the hard ceilings
	// a Server configuration may raise the limits to (LIM-ART-9).
	HistoryRevisionsCeiling int64 = 20000
	HistoryBytesCeiling     int64 = 8 << 30
	// HistoryRevisionsFloor and HistoryBytesFloor are the lowest configurable
	// limits: room for a current revision and one more revision, each of up
	// to the generic payload limit (LIM-ART-4).
	HistoryRevisionsFloor int64 = 2
	HistoryBytesFloor     int64 = 2 * MaxPayloadSize

	// MaxHistoryCandidatesPerTransaction bounds the history candidate rows
	// that one database transaction examines, and
	// MaxHistoryRemovalsPerTransaction the revision rows it removes, across
	// every binding the transaction admits (LIM-ART-10).
	MaxHistoryCandidatesPerTransaction = 64
	MaxHistoryRemovalsPerTransaction   = 64
	// MaxHistoryReclaimBytesPerPass bounds the encoded payload bytes of the
	// revisions one pass removes, and so the versions and blobs it can
	// reclaim; the first removal of a pass always goes (LIM-ART-11).
	MaxHistoryReclaimBytesPerPass int64 = 64 << 20
	// HistoryRetryAfter is the Retry-After of
	// artifact_history_maintenance_pending: short enough that three attempts
	// fit an Artifact operation deadline (ART-R127).
	HistoryRetryAfter = time.Second
)

// HistoryPolicy is the retained-history limit of every binding. The zero
// value is invalid; use DefaultHistoryPolicy or NewHistoryPolicy.
type HistoryPolicy struct {
	maxRevisions int64
	maxBytes     int64
}

// DefaultHistoryPolicy returns the documented defaults.
func DefaultHistoryPolicy() HistoryPolicy {
	return HistoryPolicy{maxRevisions: DefaultHistoryMaxRevisions, maxBytes: DefaultHistoryMaxBytes}
}

// ErrInvalidHistoryPolicy reports configured limits outside their floors and
// hard ceilings.
var ErrInvalidHistoryPolicy = errors.New("artifact history limits are outside their allowed range")

// NewHistoryPolicy validates configured limits against their floors and hard
// ceilings.
func NewHistoryPolicy(maxRevisions, maxBytes int64) (HistoryPolicy, error) {
	if maxRevisions < HistoryRevisionsFloor || maxRevisions > HistoryRevisionsCeiling {
		return HistoryPolicy{}, fmt.Errorf("%w: revisions %d not in %d..%d",
			ErrInvalidHistoryPolicy, maxRevisions, HistoryRevisionsFloor, HistoryRevisionsCeiling)
	}
	if maxBytes < HistoryBytesFloor || maxBytes > HistoryBytesCeiling {
		return HistoryPolicy{}, fmt.Errorf("%w: bytes %d not in %d..%d",
			ErrInvalidHistoryPolicy, maxBytes, HistoryBytesFloor, HistoryBytesCeiling)
	}
	return HistoryPolicy{maxRevisions: maxRevisions, maxBytes: maxBytes}, nil
}

// MaxRevisions returns the retained revisions one binding may hold.
func (p HistoryPolicy) MaxRevisions() int64 { return p.maxRevisions }

// MaxBytes returns the retained full encoded bytes one binding may hold.
func (p HistoryPolicy) MaxBytes() int64 { return p.maxBytes }

type historyPolicyKey struct{}

// WithHistoryPolicy installs the Server's configured policy in ctx. Server
// composition installs it once, beside the blob runtime, so every
// transaction-scoped repository applies the same policy.
func WithHistoryPolicy(ctx context.Context, policy HistoryPolicy) context.Context {
	return context.WithValue(ctx, historyPolicyKey{}, policy)
}

// historyPolicy returns the installed policy, or the defaults when none is
// installed, which is the policy of a Server without configured limits.
func historyPolicy(ctx context.Context) HistoryPolicy {
	if policy, ok := ctx.Value(historyPolicyKey{}).(HistoryPolicy); ok && policy.maxRevisions > 0 {
		return policy
	}
	return DefaultHistoryPolicy()
}

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
