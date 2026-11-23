//go:build integration

package findingintake

import (
	"errors"
	"testing"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// A receipt that a stored bound refuses, here the comparison projection's
// size measured on its jsonb text, is the intake's ErrInvalid with the
// PostgreSQL cause kept, which the private Artifact API answers 400 and
// non-retryable instead of a retryable 500 (PER-R13). Nothing is written.
// The bound is lowered below every projection; its name and measure are the
// production ones.
func TestPostgresReceiptRefusedByAStoredBoundIsInvalid(t *testing.T) {
	f := newDeletionImportFixture(t)
	if _, err := f.pool.Exec(f.ctx, `
ALTER TABLE finding_proposal_receipts DROP CONSTRAINT finding_comparison_projection_shape,
ADD CONSTRAINT finding_comparison_projection_shape CHECK (octet_length(comparison_projection::text) <= 2) NOT VALID`); err != nil {
		t.Fatal(err)
	}
	err := tryInsertInboxReceipt(t, f, f.request.OwnerID, f.request.RunID, "", "bounded")
	if !errors.Is(err, ErrInvalid) || persistencepostgres.Classify(err) != persistencepostgres.ClassInvalid ||
		persistencepostgres.ConstraintName(err) != "finding_comparison_projection_shape" {
		t.Fatalf("receipt over the stored bound = %v, want ErrInvalid wrapping finding_comparison_projection_shape", err)
	}
	if _, err := readReceiptByID(f.ctx, f.pool, "bounded-receipt"); err == nil {
		t.Fatal("a refused receipt was written")
	}
}
