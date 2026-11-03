package pgstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// testStore joins every seam of the Audit store, so a test can drive one
// Audit through all of them, as the Server does through several callers.
type testStore struct {
	*ClaimStore
	*OwnerStore
	*ReviewStore
	*QueryStore
	*FindingEvents
	*RunDeletion
	*ProjectDeletion
	*CredentialHolds
}

func newTestStore(db persistencepostgres.DBTX) *testStore {
	return &testStore{
		ClaimStore: NewClaimStore(db, artifactpgtest.NewStorage()).WithPurgeDependencies(purgeTestAuditHolds), OwnerStore: NewOwnerStore(db, artifactpgtest.NewStorage()), ReviewStore: NewReviewStore(db, artifactpgtest.NewStorage()),
		QueryStore: NewQueryStore(db, artifactpgtest.NewStorage()), FindingEvents: NewFindingEvents(db, artifactpgtest.NewStorage()), RunDeletion: NewRunDeletion(db, artifactpgtest.NewStorage()),
		ProjectDeletion: NewProjectDeletion(db, artifactpgtest.NewStorage()), CredentialHolds: NewCredentialHolds(db, artifactpgtest.NewStorage()),
	}
}

// This test-only composition releases the same bounded rows as finding
// intake, without importing that client back into its own store's fixtures.
func purgeTestAuditHolds(ctx context.Context, tx pgx.Tx, auditID string, limit int) (int, error) {
	tag, err := tx.Exec(ctx, `WITH batch AS (
 SELECT ctid FROM finding_proposal_audit_holds WHERE audit_id = $1 LIMIT $2
 ) DELETE FROM finding_proposal_audit_holds AS hold USING batch WHERE hold.ctid = batch.ctid`, auditID, limit)
	return int(tag.RowsAffected()), err
}
