// Package settingsstore persists Contractor's narrow, deployment-wide
// settings resources. It intentionally does not expose a generic key/value or
// JSON settings bag.
package settingsstore

import (
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
)

const (
	MinimumConcurrentRuns = 1
	MaximumConcurrentRuns = 32
	// The model-free lane pool has its own setting with the same range
	// (LIM-OPS-34).
	MinimumConcurrentToolRuns = 1
	MaximumConcurrentToolRuns = 32
	// MaximumLanes bounds the lanes of both Scheduler pools together.
	MaximumLanes = MaximumConcurrentRuns + MaximumConcurrentToolRuns
)

var (
	ErrInvalid      = apierror.New(apierror.InvalidRequest, "Scheduler settings request is invalid")
	ErrPrecondition = apierror.New(apierror.PreconditionFailed, "Scheduler settings revision precondition failed")
	ErrInvariant    = errors.New("Scheduler settings storage invariant failed")
)

// SchedulerSettings is the single PostgreSQL-authoritative Workflow Scheduler
// configuration. MaxConcurrentRuns sizes the lane pool of model Stages and
// MaxConcurrentToolRuns the pool of model-free Stages. Revision is never zero
// and UpdatedAt is normalized to UTC.
type SchedulerSettings struct {
	MaxConcurrentRuns     int
	MaxConcurrentToolRuns int
	Revision              uint64
	UpdatedAt             time.Time
}

// UpdateSchedulerSettingsParams replaces MaxConcurrentRuns and, when it is not
// nil, MaxConcurrentToolRuns; nil keeps the stored value.
type UpdateSchedulerSettingsParams struct {
	MaxConcurrentRuns     int
	MaxConcurrentToolRuns *int
	ExpectedRevision      uint64
}
