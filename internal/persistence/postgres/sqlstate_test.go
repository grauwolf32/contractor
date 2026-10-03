package postgres

import (
	"errors"
	"fmt"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestConstraintErrorClassifiesSQLStates(t *testing.T) {
	conflict, invalid := errors.New("conflict"), errors.New("invalid")
	for code, want := range map[string]error{
		SQLStateUniqueViolation:           conflict,
		SQLStateForeignKeyViolation:       invalid,
		SQLStateCheckViolation:            invalid,
		SQLStateStringDataRightTruncation: invalid,
		SQLStateInvalidTextRepresentation: invalid,
		SQLStateSerializationFailure:      nil,
	} {
		err := fmt.Errorf("write: %w", &pgconn.PgError{Code: code})
		if got := ConstraintError(err, conflict, invalid); got != want {
			t.Errorf("ConstraintError(%s) = %v, want %v", code, got, want)
		}
	}
	if got := ConstraintError(errors.New("network"), conflict, invalid); got != nil {
		t.Fatalf("ConstraintError(non-PostgreSQL) = %v", got)
	}
}
