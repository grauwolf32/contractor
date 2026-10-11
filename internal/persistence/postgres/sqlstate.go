package postgres

import (
	"context"
	"errors"
	"io"
	"net"

	"github.com/jackc/pgx/v5/pgconn"
)

// SQLSTATE codes that Contractor names. Classify is the one place that turns
// them into outcomes; other packages compare Classify results, and a source
// test rejects SQLSTATE literals anywhere else in the Go code.
const (
	SQLStateConnectionFailure     = "08006"
	SQLStateForeignKeyViolation   = "23503"
	SQLStateUniqueViolation       = "23505"
	SQLStateCheckViolation        = "23514"
	SQLStateExclusionViolation    = "23P01"
	SQLStateSerializationFailure  = "40001"
	SQLStateDeadlockDetected      = "40P01"
	SQLStateInsufficientPrivilege = "42501"
	SQLStateUndefinedColumn       = "42703"
	SQLStateUndefinedTable        = "42P01"
	SQLStateTooManyConnections    = "53300"
	// SQLStateObjectNotInPrerequisiteState is what the schema's deletion
	// fences raise when a write targets a Project that is being deleted.
	SQLStateObjectNotInPrerequisiteState = "55000"
	SQLStateLockNotAvailable             = "55P03"
	SQLStateQueryCanceled                = "57014"
	// SQLStateAssertFailure is what the schema's own invariant checks, such
	// as the Audit event allocator, raise when a statement misuses them.
	SQLStateAssertFailure = "P0004"
)

// Class is the one classification of a failed database call (PER-R13). Each
// package derives its own domain error from the Class; none interprets an
// SQLSTATE itself.
type Class uint8

const (
	// ClassNone: no error, or an error that carries neither a PostgreSQL
	// condition nor a database transport failure (pgx.ErrNoRows, a domain
	// sentinel). The caller keeps its own handling.
	ClassNone Class = iota
	// ClassDuplicate: a unique or exclusion violation. The conflicting row
	// exists; permanent for this request, which the caller replays or rejects.
	ClassDuplicate
	// ClassReferenceViolation: a foreign-key relationship does not hold: the
	// referenced row is missing, or a deleted row is still referenced. Some
	// triggers raise it for a missing parent row. Permanent.
	ClassReferenceViolation
	// ClassInvalid: the data violates a declared constraint: a not-null or
	// named CHECK constraint, or a data exception. Permanent.
	ClassInvalid
	// ClassRejected: a trigger rejected the mutation under a business rule.
	// Triggers raise check_violation without naming a constraint, which is
	// what separates them from a CHECK constraint. Permanent for the state
	// the statement observed.
	ClassRejected
	// ClassDeleting: a deletion fence rejected a write because the owning
	// Project is being deleted. Permanent.
	ClassDeleting
	// ClassPermissionDenied: the database role lacks a privilege. Permanent.
	ClassPermissionDenied
	// ClassTransactionConflict: PostgreSQL aborted the transaction on a
	// serialization failure or deadlock. Nothing committed, and a replay of the
	// whole transaction may succeed.
	ClassTransactionConflict
	// ClassTimeout: a lock wait, statement or caller deadline expired.
	// Retryable by an idempotent callee; a timeout at commit is ambiguous.
	ClassTimeout
	// ClassUnavailable: the connection or the server failed: a connection
	// exception, insufficient resources, operator intervention, a system error
	// or a lost transport. Retryable by an idempotent callee; ambiguous at
	// commit.
	ClassUnavailable
	// ClassCancelled: the caller's context was cancelled. Not retried.
	ClassCancelled
	// ClassInternal: any other PostgreSQL condition, such as a syntax,
	// undefined-object, PL/pgSQL or internal error. Permanent.
	ClassInternal
)

// sqlStateClasses maps an SQLSTATE class, the first two characters of a code,
// to its outcome when sqlStateCodes names no code-specific one. Every other
// class is ClassInternal.
var sqlStateClasses = map[string]Class{
	"08": ClassUnavailable, // connection exception
	"22": ClassInvalid,     // data exception
	"23": ClassInvalid,     // integrity constraint violation
	"40": ClassUnavailable, // transaction rollback that is not a definite conflict
	"53": ClassUnavailable, // insufficient resources
	"57": ClassUnavailable, // operator intervention
	"58": ClassUnavailable, // system error
}

// sqlStateCodes is the outcome of every code named above, whether or not it
// differs from its class's outcome; a code that is not named takes its
// class's. A check_violation is resolved by classifySQLState, not here.
var sqlStateCodes = map[string]Class{
	SQLStateConnectionFailure:            ClassUnavailable,
	SQLStateUniqueViolation:              ClassDuplicate,
	SQLStateExclusionViolation:           ClassDuplicate,
	SQLStateForeignKeyViolation:          ClassReferenceViolation,
	SQLStateSerializationFailure:         ClassTransactionConflict,
	SQLStateDeadlockDetected:             ClassTransactionConflict,
	SQLStateInsufficientPrivilege:        ClassPermissionDenied,
	SQLStateUndefinedColumn:              ClassInternal,
	SQLStateUndefinedTable:               ClassInternal,
	SQLStateTooManyConnections:           ClassUnavailable,
	SQLStateObjectNotInPrerequisiteState: ClassDeleting,
	SQLStateLockNotAvailable:             ClassTimeout,
	SQLStateQueryCanceled:                ClassTimeout,
	SQLStateAssertFailure:                ClassInternal,
}

// Classify returns the outcome class of err. A cancelled or expired caller
// context wins over any PostgreSQL condition in the same chain, because the
// condition may be the cancellation's own symptom.
func Classify(err error) Class {
	switch {
	case err == nil:
		return ClassNone
	case errors.Is(err, context.Canceled):
		return ClassCancelled
	case errors.Is(err, context.DeadlineExceeded):
		return ClassTimeout
	}
	var postgresError *pgconn.PgError
	if errors.As(err, &postgresError) {
		return classifySQLState(postgresError.Code, postgresError.ConstraintName)
	}
	var networkError net.Error
	var connectError *pgconn.ConnectError
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, net.ErrClosed) ||
		errors.As(err, &networkError) || errors.As(err, &connectError) {
		return ClassUnavailable
	}
	return ClassNone
}

func classifySQLState(code, constraint string) Class {
	if code == SQLStateCheckViolation {
		if constraint == "" {
			return ClassRejected
		}
		return ClassInvalid
	}
	if class, ok := sqlStateCodes[code]; ok {
		return class
	}
	if len(code) == 5 {
		if class, ok := sqlStateClasses[code[:2]]; ok {
			return class
		}
	}
	return ClassInternal
}

// ConstraintError maps a duplicate to conflict and a reference, constraint
// or business-rule violation to invalid. It returns nil for every other class
// so callers keep their own fallback.
func ConstraintError(err error, conflict, invalid error) error {
	switch Classify(err) {
	case ClassDuplicate:
		return conflict
	case ClassReferenceViolation, ClassInvalid, ClassRejected:
		return invalid
	default:
		return nil
	}
}
