// Package projectstore owns durable owner-scoped Project metadata. A Project
// groups reusable artifacts and Runs; it is not an execution state machine.
package projectstore

import "errors"

var (
	ErrInvalid  = errors.New("projectstore invalid argument")
	ErrNotFound = errors.New("projectstore resource not found")
	// ErrConflict is an identity conflict; a reused creation key is
	// contracts.ErrIdempotencyConflict.
	ErrConflict     = errors.New("projectstore identity conflict")
	ErrPrecondition = errors.New("projectstore revision precondition failed")
	ErrDeleting     = errors.New("projectstore Project is deleting")
	// ErrDeletionClaimLost reports that a Project no longer holds a deletion
	// claim in the claim's phase: the claim expired and another controller
	// took it, or the Project moved on.
	ErrDeletionClaimLost = errors.New("projectstore Project deletion claim was lost")
)
