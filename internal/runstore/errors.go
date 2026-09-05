// Package runstore defines durable WorkflowRun, StageExecution, and Planner
// session repositories independently from Scheduler and HTTP concerns.
package runstore

import (
	"errors"
	"fmt"
)

var (
	ErrInvalid      = errors.New("runstore invalid argument")
	ErrNotFound     = errors.New("runstore resource not found")
	ErrConflict     = errors.New("runstore optimistic state conflict")
	ErrNoWork       = errors.New("runstore has no claimable work")
	ErrPrecondition = errors.New("runstore revision precondition failed")
	ErrQueuePaused  = errors.New("owner WorkflowRun queue is paused")
)

type RunNotDeletableReason string

const (
	RunNotTerminal              RunNotDeletableReason = "run_not_terminal"
	RunAllocationReleasePending RunNotDeletableReason = "allocation_release_pending"
)

// RunNotDeletableError identifies the lifecycle gate that rejected a hard
// deletion without exposing any owner information.
type RunNotDeletableError struct {
	RunID  string
	Reason RunNotDeletableReason
}

func (e *RunNotDeletableError) Error() string {
	return fmt.Sprintf("WorkflowRun %q is not deletable: %s", e.RunID, e.Reason)
}

func (e *RunNotDeletableError) Unwrap() error { return ErrConflict }

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
