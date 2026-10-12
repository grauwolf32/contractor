// Package projectstore owns durable owner-scoped Project metadata. A Project
// groups reusable artifacts and Runs; it is not an execution state machine.
package projectstore

import "github.com/grauwolf32/contractor/internal/apierror"

var (
	ErrInvalid  = apierror.New(apierror.InvalidRequest, "projectstore invalid argument")
	ErrNotFound = apierror.New(apierror.NotFound, "projectstore resource not found")
	// ErrConflict is an identity conflict; a reused creation key is
	// contracts.ErrIdempotencyConflict.
	ErrConflict     = apierror.New(apierror.Conflict, "projectstore identity conflict")
	ErrPrecondition = apierror.New(apierror.PreconditionFailed, "projectstore revision precondition failed")
	ErrDeleting     = apierror.New(apierror.ProjectDeleting, "projectstore Project is deleting")
)
