package findingintake_test

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/rundeletion"
)

// Run deletion composes findingintake.MarkSourceRunDeleted, so only this
// external test package can import internal/rundeletion. It lends the Run
// deleter to the package's own tests before they run.
func init() {
	findingintake.DeleteRunForTest = func(
		ctx context.Context, db persistencepostgres.DBTX, ownerID, runID string,
	) error {
		return rundeletion.New(db).DeleteReleasedTerminalRun(ctx, ownerID, runID)
	}
}

// TestRunDeletionIsLentToThePackageTests keeps this package in the test
// binary: go test links an external test package only when it has a test, and
// without this package the tests that delete Runs would find no deleter.
func TestRunDeletionIsLentToThePackageTests(t *testing.T) {
	if findingintake.DeleteRunForTest == nil {
		t.Fatal("the findingintake tests have no Run deleter")
	}
}
