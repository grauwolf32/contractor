package postgres

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestSafeErrorPreservesSQLStateAndCancellationWithoutRenderingCause(t *testing.T) {
	for _, cause := range []error{&pgconn.PgError{Code: "40001", Message: "private-sql-payload"}, context.Canceled} {
		err := WrapError("safe operation", cause)
		if !errors.Is(err, cause) || strings.Contains(fmt.Sprintf("%+v", err), "private") {
			t.Fatalf("cause identity or redaction failed: %v", err)
		}
		if cause != context.Canceled && SQLState(err) != "40001" {
			t.Fatal("SQLSTATE was lost")
		}
	}
}

func TestTransactionRetryOnlyReplaysDefiniteConflicts(t *testing.T) {
	for _, test := range []struct {
		name     string
		err      error
		attempts int
	}{
		{"serialization", &pgconn.PgError{Code: "40001"}, 3},
		{"deadlock", &pgconn.PgError{Code: "40P01"}, 3},
		{"business-cas", errors.New("precondition failed"), 1},
		{"unique", &pgconn.PgError{Code: "23505"}, 1},
		{"ambiguous-commit", WrapError("commit PostgreSQL transaction", io.ErrUnexpectedEOF), 1},
		{"cancelled", errors.Join(context.Canceled, &pgconn.PgError{Code: "40001"}), 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			calls := 0
			err := retryTransaction(t.Context(), func() error { calls++; return test.err })
			if !errors.Is(err, test.err) || calls != test.attempts {
				t.Fatalf("retry = %d attempts, error %v", calls, err)
			}
		})
	}
}

func TestTransactionRetryStopsAfterSuccessAndDuringBackoffCancellation(t *testing.T) {
	calls := 0
	if err := retryTransaction(t.Context(), func() error {
		calls++
		if calls == 1 {
			return &pgconn.PgError{Code: "40001"}
		}
		return nil
	}); err != nil || calls != 2 {
		t.Fatalf("success = %d, %v", calls, err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	calls = 0
	err := retryTransaction(ctx, func() error {
		calls++
		cancel()
		return &pgconn.PgError{Code: "40001"}
	})
	if !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatalf("cancel = %d, %v", calls, err)
	}
}
