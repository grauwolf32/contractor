package public

import (
	"context"
	"errors"
	"io/fs"
	"net"
	"net/http"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/requestid"
)

// This wrapper is inside authentication so it observes the same Request that
// ServeMux annotates. Pattern is a registered route template, not the raw path
// (which may contain Artifact names, credentials or other user input).
func withErrorDiagnostics(mux *http.ServeMux) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mux.ServeHTTP(w, r)
		requestid.AnnotateFailure(w, r.Pattern, "unknown")
	})
}

// diagnosticCause is intentionally a closed vocabulary. It inspects typed
// causes and sentinel identities only: neither Error(), SQL server messages,
// network addresses nor wrapped provider payloads are suitable log values.
// Classification never affects the public status, envelope or retry policy.
func diagnosticCause(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "cancelled"
	case errors.Is(err, context.DeadlineExceeded):
		return "deadline_exceeded"
	case errors.Is(err, credentials.ErrGatewayUnavailable), errors.Is(err, credentials.ErrManagerUnavailable):
		return "downstream_unavailable"
	case errors.Is(err, credentials.ErrKeyUnavailable):
		return "credential_key_unavailable"
	case errors.Is(err, credentials.ErrCrypto):
		return "credential_crypto_failed"
	case errors.Is(err, pgx.ErrTxClosed), errors.Is(err, pgx.ErrTxCommitRollback):
		return "storage_transaction_failed"
	}
	var postgresError *pgconn.PgError
	if errors.As(err, &postgresError) {
		switch postgresError.Code {
		case "40001":
			return "storage_serialization_conflict"
		case "40P01":
			return "storage_deadlock"
		case "55P03":
			return "storage_lock_unavailable"
		case "57014":
			return "storage_query_cancelled"
		case "23502", "23503", "23505", "23514", "23P01":
			return "storage_constraint_failed"
		case "53300", "57P01", "57P02", "57P03", "08000", "08001", "08003", "08004", "08006", "08007", "08P01":
			return "storage_unavailable"
		default:
			return "storage_failed"
		}
	}
	var connectionError *pgconn.ConnectError
	if errors.As(err, &connectionError) {
		return "storage_unavailable"
	}
	var networkError net.Error
	if errors.As(err, &networkError) {
		if networkError.Timeout() {
			return "network_timeout"
		}
		return "network_failed"
	}
	var pathError *fs.PathError
	if errors.As(err, &pathError) {
		return "storage_io_failed"
	}
	return "unknown"
}
