// Package auditstore owns the durable Audit controller state. It deliberately
// does not create WorkflowRuns, parse Worker result packages, or expose HTTP.
package auditstore

import (
	"errors"
	"fmt"
)

var (
	ErrInvalid         = errors.New("auditstore invalid argument")
	ErrNotFound        = errors.New("auditstore resource not found")
	ErrConflict        = errors.New("auditstore identity or idempotency conflict")
	ErrPrecondition    = errors.New("auditstore revision or state precondition failed")
	ErrProjectDeleting = errors.New("auditstore Project is deleting")
	ErrClaimLost       = errors.New("auditstore Controller claim is stale")
	ErrNoWork          = errors.New("auditstore has no claimable work")
)

func invalidf(format string, arguments ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalid, fmt.Sprintf(format, arguments...))
}
