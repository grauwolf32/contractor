package performance

import (
	"context"
	"errors"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestDatabaseReasonPreservesClosedConnectionAndContextCauses(t *testing.T) {
	for _, test := range []struct {
		name string
		err  error
		want Reason
	}{
		{"closed", pgconn.ErrConnClosed, DatabaseUnavailable},
		{"wrapped closed", persistencepostgres.WrapError("read diagnostics", pgconn.ErrConnClosed), DatabaseUnavailable},
		{"closed and cancelled", errors.Join(pgconn.ErrConnClosed, context.Canceled), BudgetExceeded},
		{"closed and expired", errors.Join(pgconn.ErrConnClosed, context.DeadlineExceeded), BudgetExceeded},
		{"permission", &pgconn.PgError{Code: persistencepostgres.SQLStateInsufficientPrivilege}, PermissionDenied},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := databaseReason(test.err); got != test.want {
				t.Fatalf("reason = %q, want %q", got, test.want)
			}
		})
	}
}
