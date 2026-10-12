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
)

var (
	ErrInvalid      = apierror.New(apierror.InvalidRequest, "Scheduler settings request is invalid")
	ErrPrecondition = apierror.New(apierror.PreconditionFailed, "Scheduler settings revision precondition failed")
	ErrInvariant    = errors.New("Scheduler settings storage invariant failed")
)

// SchedulerSettings is the single PostgreSQL-authoritative Workflow Scheduler
// configuration. Revision is never zero and UpdatedAt is normalized to UTC.
type SchedulerSettings struct {
	MaxConcurrentRuns int
	Revision          uint64
	UpdatedAt         time.Time
}

type UpdateSchedulerSettingsParams struct {
	MaxConcurrentRuns int
	ExpectedRevision  uint64
}
