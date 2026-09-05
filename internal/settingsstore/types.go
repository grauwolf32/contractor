// Package settingsstore persists Contractor's narrow, deployment-wide
// settings resources. It intentionally does not expose a generic key/value or
// JSON settings bag.
package settingsstore

import (
	"errors"
	"time"
)

const (
	MinimumConcurrentRuns = 1
	MaximumConcurrentRuns = 32
)

var (
	ErrInvalid      = errors.New("Scheduler settings request is invalid")
	ErrPrecondition = errors.New("Scheduler settings revision precondition failed")
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
