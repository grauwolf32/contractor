// Package auditstore is the vocabulary of the durable Audit state: Audits,
// Rounds, items, executions, receipts, reviews and reports, their states and
// limits, the parameters of each store operation and the errors it returns.
// The store itself is split by seam, and a caller imports only the seam it
// works in: claimstore (the commands the Audit controller runs under its
// claim), ownerstore (the owner's lifecycle commands), reviewstore (the review
// journal), querystore (the reads) and hookstore (the entry points of other
// contexts). The Audit store deliberately does not create WorkflowRuns, parse
// Worker result packages, or expose HTTP.
package auditstore

import (
	"errors"
	"fmt"
)

var (
	ErrInvalid  = errors.New("auditstore invalid argument")
	ErrNotFound = errors.New("auditstore resource not found")
	// ErrConflict is an identity, state or internal submission-replay conflict;
	// a reused public Idempotency-Key is contracts.ErrIdempotencyConflict.
	ErrConflict        = errors.New("auditstore identity or submission conflict")
	ErrPrecondition    = errors.New("auditstore revision or state precondition failed")
	ErrProjectDeleting = errors.New("auditstore Project is deleting")
	ErrClaimLost       = errors.New("auditstore Controller claim is stale")
	// ErrEvidenceBudgetExhausted is the collection precondition failure caused
	// by the retained evidence budget. The budget only grows, so the same
	// collection request can never succeed and needs a terminal receipt.
	ErrEvidenceBudgetExhausted = fmt.Errorf("%w: Audit evidence budget is exhausted", ErrPrecondition)
)
