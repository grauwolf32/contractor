// Package projectstore owns durable owner-scoped Project metadata. A Project
// groups reusable artifacts and Runs; it is not an execution state machine.
package projectstore

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/apierror"
)

var (
	ErrInvalid  = apierror.New(apierror.InvalidRequest, "projectstore invalid argument")
	ErrNotFound = apierror.New(apierror.NotFound, "projectstore resource not found")
	// ErrConflict is an identity conflict; a reused creation key is
	// contracts.ErrIdempotencyConflict.
	ErrConflict     = apierror.New(apierror.Conflict, "projectstore identity conflict")
	ErrPrecondition = apierror.New(apierror.PreconditionFailed, "projectstore revision precondition failed")
	ErrDeleting     = apierror.New(apierror.ProjectDeleting, "projectstore Project is deleting")
	// ErrDeletionClaimLost reports that a Project no longer holds a deletion
	// claim in the claim's phase: the claim expired and another controller
	// took it, or the Project moved on. It has no public Kind: only the
	// Project deletion controller holds claims, and no request answers it.
	ErrDeletionClaimLost = errors.New("projectstore Project deletion claim was lost")
)
