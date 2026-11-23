package pgstore

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func invalidf(format string, arguments ...any) error {
	return fmt.Errorf("%w: %s", auditstore.ErrInvalid, fmt.Sprintf(format, arguments...))
}

// writeError returns a failed write of operation. A named CHECK constraint
// or another integrity rule that refused the written values is the store's
// ErrInvalid, with the PostgreSQL cause kept (PER-R13): such a rule, for
// example a jsonb size bound measured on the re-serialised text, can refuse
// values that the Go validation accepted, and the same write is refused
// again. Every other failure keeps its own class.
func writeError(operation string, err error) error {
	if persistencepostgres.Classify(err) == persistencepostgres.ClassInvalid {
		return fmt.Errorf("%s: %w: %w", operation, auditstore.ErrInvalid, err)
	}
	return fmt.Errorf("%s: %w", operation, err)
}
