// Package runstore defines durable WorkflowRun, StageExecution, and Planner
// session repositories independently from Scheduler and HTTP concerns.
package runstore

import (
	"errors"
	"fmt"
)

var (
	ErrInvalid         = errors.New("runstore invalid argument")
	ErrNotFound        = errors.New("runstore resource not found")
	ErrConflict        = errors.New("runstore optimistic state conflict")
	ErrNoWork          = errors.New("runstore has no claimable work")
	ErrPrecondition    = errors.New("runstore revision precondition failed")
	ErrQueuePaused     = errors.New("owner WorkflowRun queue is paused")
	ErrProjectDeleting = errors.New("WorkflowRun Project is deleting")
	// ErrMultipleActiveStages reports a Run with more than one active
	// StageExecution, which SCH-R114 forbids: durable state the Scheduler
	// cannot progress.
	ErrMultipleActiveStages = errors.New("WorkflowRun has more than one active StageExecution")
)

// StateConflictError reports the compare-and-swap predicate that lost.
type StateConflictError struct {
	Resource string
	ID       string
	Expected string
}

func (e *StateConflictError) Error() string {
	return fmt.Sprintf("%s %q is not in expected state %q", e.Resource, e.ID, e.Expected)
}

func (e *StateConflictError) Unwrap() error { return ErrConflict }

func invalidf(format string, arguments ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalid, fmt.Sprintf(format, arguments...))
}
