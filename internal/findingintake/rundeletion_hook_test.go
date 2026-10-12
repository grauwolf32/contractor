//go:build integration

package findingintake

import (
	"context"
	"errors"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// DeleteRunForTest is rundeletion.New(db).DeleteReleasedTerminalRun. Run
// deletion composes MarkSourceRunDeleted, so the tests of this package cannot
// import internal/rundeletion without an import cycle; rundeletion_link_test.go,
// in the external test package, sets this before any test runs.
var DeleteRunForTest func(ctx context.Context, db persistencepostgres.DBTX, ownerID, runID string) error

// deleteRun deletes a released terminal Run through internal/rundeletion, the
// whole cascade included.
func deleteRun(ctx context.Context, db persistencepostgres.DBTX, ownerID, runID string) error {
	if DeleteRunForTest == nil {
		return errors.New("rundeletion_link_test.go did not lend the Run deleter")
	}
	return DeleteRunForTest(ctx, db, ownerID, runID)
}
