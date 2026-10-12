package pgstore

import (
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
		ClaimStore: NewClaimStore(db), OwnerStore: NewOwnerStore(db), ReviewStore: NewReviewStore(db),
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
	*FindingEvents
	*RunDeletionTx
	*ProjectDeletion
	*CredentialHolds
}

func newTestTxStore(tx pgx.Tx) *testTxStore {
	return &testTxStore{
		ClaimStore: NewClaimStore(tx), OwnerTxStore: NewOwnerTxStore(tx), ReviewTxStore: NewReviewTxStore(tx),
		QueryStore: NewQueryStore(tx), FindingEvents: NewFindingEvents(tx), RunDeletionTx: NewRunDeletionTx(tx),
		ProjectDeletion: NewProjectDeletion(tx), CredentialHolds: NewCredentialHolds(tx),
	}
}
