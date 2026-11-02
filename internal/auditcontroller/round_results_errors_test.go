package auditcontroller

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestControllerClassifiesExactInputInitializationFailures(t *testing.T) {
	permanent := []error{artifacts.ErrInvalidScope, artifacts.ErrInvalidName, artifacts.ErrExactRevisionRequired,
		artifacts.ErrArtifactNotFound, blob.ErrIntegrity, artifacts.ErrArtifactConflict, artifacts.ErrArtifactFrozen,
		artifacts.ErrHistoryProtected, artifacts.ErrReservedNamespace, artifacts.ErrRunRevisionLimit, artifacts.ErrInvalidMediaType,
		blob.ErrTooLarge, artifacts.ErrVersionedWriteTarget, contracts.ErrValidation, contracts.ErrExactArtifactMismatch,
		&auditdomain.ValidationError{Code: auditdomain.CodeLimitExceeded},
		&pgconn.PgError{Code: persistencepostgres.SQLStateForeignKeyViolation, ConstraintName: "artifact_exact_hold"}}
	transient := []error{context.Canceled, context.DeadlineExceeded, errors.New("temporary storage outage"), blob.ErrIO,
		artifacts.ErrHistoryMaintenancePending, &pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure},
		&pgconn.PgError{Code: persistencepostgres.SQLStateForeignKeyViolation, ConstraintName: "unrelated_foreign_key"}}
	for _, failure := range permanent {
		t.Run("permanent-"+failure.Error(), func(t *testing.T) {
			if !permanentSubmissionError(fmt.Errorf("prepare declared input: %w", failure)) {
				t.Fatal("permanent exact input rejection remains retryable")
			}
		})
	}
	for _, failure := range transient {
		t.Run("transient-"+failure.Error(), func(t *testing.T) {
			if permanentSubmissionError(fmt.Errorf("prepare declared input: %w", failure)) {
				t.Fatal("uncertain storage/cancellation became permanent")
			}
		})
	}
}
