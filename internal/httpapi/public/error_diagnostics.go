package public

import (
	"context"
	"errors"
	"io/fs"
	"net/http"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
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
	case errors.Is(err, credentials.ErrRecoveryRequired):
		return "credential_recovery_required"
	case errors.Is(err, blob.ErrTransferCapacity):
		return "artifact_transfer_capacity"
	case errors.Is(err, blob.ErrIntegrity):
		return "artifact_integrity_failed"
	case errors.Is(err, blob.ErrMissing):
		return "artifact_content_missing"
	case errors.Is(err, artifacts.ErrHistoryProtected):
		return "artifact_history_protected"
	case errors.Is(err, artifacts.ErrHistoryMaintenancePending):
		return "artifact_history_maintenance_pending"
	case errors.Is(err, pgx.ErrTxClosed), errors.Is(err, pgx.ErrTxCommitRollback):
		return "storage_transaction_failed"
	}
	// Explicit domain causes above retain their diagnostic; an unclassified
	// database/transport cause follows the shared persistence classification.
	if persistencepostgres.Classify(err) == persistencepostgres.ClassUnavailable {
		return "storage_unavailable"
	}
	var postgresError *pgconn.PgError
	if errors.As(err, &postgresError) {
		switch postgresError.Code {
		case persistencepostgres.SQLStateSerializationFailure:
			return "storage_serialization_conflict"
		case persistencepostgres.SQLStateDeadlockDetected:
			return "storage_deadlock"
		case persistencepostgres.SQLStateLockNotAvailable:
			return "storage_lock_unavailable"
		case persistencepostgres.SQLStateQueryCanceled:
			return "storage_query_cancelled"
		}
		switch persistencepostgres.Classify(err) {
		case persistencepostgres.ClassDuplicate, persistencepostgres.ClassReferenceViolation,
			persistencepostgres.ClassInvalid, persistencepostgres.ClassRejected,
			persistencepostgres.ClassLimitExceeded:
			return "storage_constraint_failed"
		default:
			return "storage_failed"
		}
	}
	var pathError *fs.PathError
	if errors.As(err, &pathError) {
		return "storage_io_failed"
	}
	return "unknown"
}
