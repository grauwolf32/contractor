package postgres

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestTransientFailureClassifiesWaitsAndLostConnections(t *testing.T) {
	for _, test := range []struct {
		name string
		err  error
		want bool
	}{
		{"serialization", &pgconn.PgError{Code: SQLStateSerializationFailure}, true},
		{"deadlock", &pgconn.PgError{Code: SQLStateDeadlockDetected}, true},
		{"connection-exhaustion", &pgconn.PgError{Code: SQLStateTooManyConnections}, true},
		{"lock-timeout", WrapError("lock Run", &pgconn.PgError{Code: SQLStateLockNotAvailable}), true},
		{"statement-timeout", &pgconn.PgError{Code: SQLStateQueryCanceled}, true},
		{"acquire-timeout", WrapError("begin PostgreSQL transaction", context.DeadlineExceeded), true},
		{"lost-connection", fmt.Errorf("read: %w", io.ErrUnexpectedEOF), true},
		{"closed-connection", net.ErrClosed, true},
		{"network-timeout", &net.OpError{Op: "read", Err: os.ErrDeadlineExceeded}, true},
		{"unique", &pgconn.PgError{Code: SQLStateUniqueViolation}, false},
		{"privilege", &pgconn.PgError{Code: SQLStateInsufficientPrivilege}, false},
		{"cancelled", context.Canceled, false},
		{"business", errors.New("precondition failed"), false},
	} {
		if got := IsTransientFailure(test.err); got != test.want {
			t.Errorf("IsTransientFailure(%s) = %t, want %t", test.name, got, test.want)
		}
	}
}
