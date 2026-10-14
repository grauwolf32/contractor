package pgstore

import (
	"context"
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
		ClaimStore: NewClaimStore(db).WithPurgeDependencies(purgeTestAuditHolds), OwnerStore: NewOwnerStore(db), ReviewStore: NewReviewStore(db),
		QueryStore: NewQueryStore(db), FindingEvents: NewFindingEvents(db), RunDeletion: NewRunDeletion(db),
		ProjectDeletion: NewProjectDeletion(db), CredentialHolds: NewCredentialHolds(db),
	}
}

// testTxStore is a testStore bound to the caller's transaction, with the
// transaction-bound steps of every seam.
type testTxStore struct {
	*ClaimStore
	*OwnerTxStore
	*ReviewTxStore
	*QueryStore
	*FindingEventsTx
	*RunDeletionTx
	*ProjectDeletion
	*CredentialHolds
}

func newTestTxStore(tx pgx.Tx) *testTxStore {
	return &testTxStore{
		ClaimStore: NewClaimStore(tx).WithPurgeDependencies(purgeTestAuditHolds), OwnerTxStore: NewOwnerTxStore(tx), ReviewTxStore: NewReviewTxStore(tx),
		QueryStore: NewQueryStore(tx), FindingEventsTx: NewFindingEventsTx(tx), RunDeletionTx: NewRunDeletionTx(tx),
		ProjectDeletion: NewProjectDeletion(tx), CredentialHolds: NewCredentialHolds(tx),
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
