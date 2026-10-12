package postgres

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// TestClassifyMapsEverySQLStateClassToOneOutcome is the classification table
// of PER-R13. The literals here are the independent oracle for the named
// constants, so this is the only test that spells SQLSTATE codes.
func TestClassifyMapsEverySQLStateClassToOneOutcome(t *testing.T) {
	// Every class of PostgreSQL's SQLSTATE appendix, by its generic code.
	classes := map[string]Class{
		"00000": ClassInternal, "01000": ClassInternal, "02000": ClassInternal, "03000": ClassInternal,
		"08000": ClassUnavailable, "09000": ClassInternal, "0A000": ClassInternal, "0B000": ClassInternal,
		"0F000": ClassInternal, "0L000": ClassInternal, "0P000": ClassInternal, "0Z000": ClassInternal,
		"20000": ClassInternal, "21000": ClassInternal, "22000": ClassInvalid, "23000": ClassInvalid,
		"24000": ClassInternal, "25000": ClassInternal, "26000": ClassInternal, "27000": ClassInternal,
		"28000": ClassInternal, "2B000": ClassInternal, "2D000": ClassInternal, "2F000": ClassInternal,
		"34000": ClassInternal, "38000": ClassInternal, "39000": ClassInternal, "3B000": ClassInternal,
		"3D000": ClassInternal, "3F000": ClassInternal, "40000": ClassUnavailable, "42000": ClassInternal,
		"44000": ClassInternal, "53000": ClassUnavailable, "54000": ClassInternal, "55000": ClassDeleting,
		"57000": ClassUnavailable, "58000": ClassUnavailable, "72000": ClassInternal, "F0000": ClassInternal,
		"HV000": ClassInternal, "P0000": ClassInternal, "XX000": ClassInternal,
	}
	// Codes whose outcome is not their class's, and representative members of
	// the classes above that callers depend on.
	codes := map[string]Class{
		"08006": ClassUnavailable, "08P01": ClassUnavailable,
		"22001": ClassInvalid, "22P02": ClassInvalid,
		"23502": ClassInvalid, "23503": ClassReferenceViolation, "23505": ClassDuplicate,
		"23P01": ClassDuplicate,
		"25P02": ClassInternal, "28P01": ClassInternal,
		"40001": ClassTransactionConflict, "40P01": ClassTransactionConflict, "40003": ClassUnavailable,
		"42501": ClassPermissionDenied, "42703": ClassInternal, "42P01": ClassInternal, "42601": ClassInternal,
		"53100": ClassUnavailable, "53300": ClassUnavailable,
		"55006": ClassInternal, "55P03": ClassTimeout,
		"57014": ClassTimeout, "57P01": ClassUnavailable, "57P02": ClassUnavailable, "57P03": ClassUnavailable,
		"58030": ClassUnavailable, "P0001": ClassInternal, "P0004": ClassInternal, "XX001": ClassInternal,
		"": ClassInternal, "4000": ClassInternal, "ZZ999": ClassInternal,
	}
	for _, table := range []map[string]Class{classes, codes} {
		for code, want := range table {
			err := fmt.Errorf("query: %w", WrapError("safe message", &pgconn.PgError{Code: code}))
			if got := Classify(err); got != want {
				t.Errorf("Classify(%q) = %d, want %d", code, got, want)
			}
		}
	}
	// Every named constant spells its code and has an explicit outcome; only
	// check_violation and program_limit_exceeded are resolved by their
	// constraint name instead.
	for constant, code := range map[string]string{
		SQLStateConnectionFailure: "08006", SQLStateForeignKeyViolation: "23503", SQLStateUniqueViolation: "23505",
		SQLStateCheckViolation: "23514", SQLStateExclusionViolation: "23P01",
		SQLStateSerializationFailure: "40001", SQLStateDeadlockDetected: "40P01",
		SQLStateInsufficientPrivilege: "42501", SQLStateUndefinedColumn: "42703", SQLStateUndefinedTable: "42P01",
		SQLStateTooManyConnections: "53300", SQLStateProgramLimitExceeded: "54000",
		SQLStateObjectNotInPrerequisiteState: "55000", SQLStateLockNotAvailable: "55P03",
		SQLStateQueryCanceled: "57014", SQLStateAssertFailure: "P0004",
	} {
		if constant != code {
			t.Errorf("SQLSTATE constant %q, want %q", constant, code)
		}
		byConstraint := constant == SQLStateCheckViolation || constant == SQLStateProgramLimitExceeded
		if _, named := sqlStateCodes[constant]; named == byConstraint {
			t.Errorf("SQLSTATE constant %q: explicit outcome = %t", constant, named)
		}
	}
	if len(sqlStateCodes) != 14 {
		t.Errorf("sqlStateCodes has %d codes; add each named constant to this test", len(sqlStateCodes))
	}
}

// A CHECK constraint names itself; a trigger's business rejection does not.
func TestClassifySeparatesCheckConstraintsFromTriggerRejections(t *testing.T) {
	named := &pgconn.PgError{Code: "23514", ConstraintName: "audits_state_check"}
	if got := Classify(named); got != ClassInvalid {
		t.Fatalf("named check violation = %d, want ClassInvalid", got)
	}
	if got := Classify(&pgconn.PgError{Code: "23514"}); got != ClassRejected {
		t.Fatalf("trigger check violation = %d, want ClassRejected", got)
	}
}

// A bound trigger names the bound it enforces; PostgreSQL's own program
// limits, such as an oversized index row, name none and stay internal.
func TestClassifySeparatesStoredBoundsFromProgramLimits(t *testing.T) {
	bound := &pgconn.PgError{Code: "54000", ConstraintName: "artifact_scopes_run_revision_limit"}
	if got := Classify(bound); got != ClassLimitExceeded {
		t.Fatalf("named program limit = %d, want ClassLimitExceeded", got)
	}
	if got := Classify(&pgconn.PgError{Code: "54000"}); got != ClassInternal {
		t.Fatalf("unnamed program limit = %d, want ClassInternal", got)
	}
}

func TestClassifyTransportAndContextFailures(t *testing.T) {
	for _, test := range []struct {
		name string
		err  error
		want Class
	}{
		{"nil", nil, ClassNone},
		{"no-rows", pgx.ErrNoRows, ClassNone},
		{"domain", errors.New("precondition failed"), ClassNone},
		{"cancelled", context.Canceled, ClassCancelled},
		{"cancelled-wins", fmt.Errorf("%w: %w", context.Canceled, &pgconn.PgError{Code: "40001"}), ClassCancelled},
		{"deadline", WrapError("begin PostgreSQL transaction", context.DeadlineExceeded), ClassTimeout},
		{"deadline-wins", fmt.Errorf("%w: %w", context.DeadlineExceeded, &pgconn.PgError{Code: "23505"}), ClassTimeout},
		{"eof", fmt.Errorf("read: %w", io.EOF), ClassUnavailable},
		{"unexpected-eof", io.ErrUnexpectedEOF, ClassUnavailable},
		{"closed", net.ErrClosed, ClassUnavailable},
		{"network-timeout", &net.OpError{Op: "read", Err: os.ErrDeadlineExceeded}, ClassUnavailable},
		{"connection-refused", &net.OpError{Op: "dial", Err: errors.New("refused")}, ClassUnavailable},
	} {
		if got := Classify(test.err); got != test.want {
			t.Errorf("Classify(%s) = %d, want %d", test.name, got, test.want)
		}
	}
}

func TestConstraintErrorClassifiesSQLStates(t *testing.T) {
	conflict, invalid := errors.New("conflict"), errors.New("invalid")
	for _, test := range []struct {
		err  *pgconn.PgError
		want error
	}{
		{&pgconn.PgError{Code: SQLStateUniqueViolation}, conflict},
		{&pgconn.PgError{Code: SQLStateExclusionViolation}, conflict},
		{&pgconn.PgError{Code: SQLStateForeignKeyViolation}, invalid},
		{&pgconn.PgError{Code: SQLStateCheckViolation, ConstraintName: "named_check"}, invalid},
		{&pgconn.PgError{Code: SQLStateCheckViolation}, invalid},
		{&pgconn.PgError{Code: "23502"}, invalid},
		{&pgconn.PgError{Code: "22001"}, invalid},
		{&pgconn.PgError{Code: "22P02"}, invalid},
		{&pgconn.PgError{Code: SQLStateSerializationFailure}, nil},
		{&pgconn.PgError{Code: SQLStateObjectNotInPrerequisiteState}, nil},
	} {
		err := fmt.Errorf("write: %w", test.err)
		if got := ConstraintError(err, conflict, invalid); got != test.want {
			t.Errorf("ConstraintError(%s %q) = %v, want %v", test.err.Code, test.err.ConstraintName, got, test.want)
		}
	}
	if got := ConstraintError(errors.New("network"), conflict, invalid); got != nil {
		t.Fatalf("ConstraintError(non-PostgreSQL) = %v", got)
	}
}
