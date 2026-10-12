// Package auditstore owns the durable Audit controller state. It deliberately
// does not create WorkflowRuns, parse Worker result packages, or expose HTTP.
package auditstore

import (
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/apierror"
)

var (
	ErrInvalid  = apierror.New(apierror.InvalidRequest, "auditstore invalid argument")
	ErrNotFound = apierror.New(apierror.NotFound, "auditstore resource not found")
	// ErrConflict is an identity, state or internal submission-replay conflict;
	// a reused public Idempotency-Key is contracts.ErrIdempotencyConflict.
	ErrConflict        = apierror.New(apierror.Conflict, "auditstore identity or submission conflict")
	ErrPrecondition    = apierror.New(apierror.PreconditionFailed, "auditstore revision or state precondition failed")
	ErrProjectDeleting = apierror.New(apierror.ProjectDeleting, "auditstore Project is deleting")
	ErrClaimLost       = errors.New("auditstore Controller claim is stale")
	// ErrEvidenceBudgetExhausted is the collection precondition failure caused
	// by the retained evidence budget. The budget only grows, so the same
	// collection request can never succeed and needs a terminal receipt.
	ErrEvidenceBudgetExhausted = fmt.Errorf("%w: Audit evidence budget is exhausted", ErrPrecondition)
)

func invalidf(format string, arguments ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalid, fmt.Sprintf(format, arguments...))
}
