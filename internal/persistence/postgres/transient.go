package postgres

import (
	"context"
	"errors"
	"io"
	"net"
)

// IsTransientFailure reports a database wait or connection failure that a
// later attempt of the same operation may outlast: a pool-acquire, lock or
// statement timeout, a serialization or deadlock abort, connection exhaustion
// or a lost connection. It does not prove that the failed attempt had no
// effect, because a timeout or lost connection can follow a commit. Only an
// idempotent callee, which accepts its own already committed outcome, may be
// retried on it.
func IsTransientFailure(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, io.EOF) ||
		errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, net.ErrClosed) {
		return true
	}
	var networkError net.Error
	if errors.As(err, &networkError) && (networkError.Timeout() || networkError.Temporary()) {
		return true
	}
	switch SQLState(err) {
	case SQLStateSerializationFailure, SQLStateDeadlockDetected, SQLStateTooManyConnections,
		SQLStateLockNotAvailable, SQLStateQueryCanceled:
		return true
	default:
		return false
	}
}
