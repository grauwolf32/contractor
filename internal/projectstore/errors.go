// Package projectstore owns durable owner-scoped Project metadata. A Project
// groups reusable artifacts and Runs; it is not an execution state machine.
package projectstore

import "errors"

var (
	ErrInvalid      = errors.New("projectstore invalid argument")
	ErrNotFound     = errors.New("projectstore resource not found")
	ErrConflict     = errors.New("projectstore identity or idempotency conflict")
	ErrPrecondition = errors.New("projectstore revision precondition failed")
)
