// Package auditstoretest gives tests one Audit store with every seam, so a
// test can set up and inspect an Audit across the seams the Server reaches
// through several callers. Production code holds only the seam it uses;
// internal/archtest keeps this package out of every non-test package.
package auditstoretest

import (
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/internal/pgstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// Store joins every seam of the Audit store on one database handle.
type Store struct {
	*pgstore.ClaimStore
	*pgstore.OwnerStore
	*pgstore.ReviewStore
	*pgstore.QueryStore
	*pgstore.FindingEvents
	*pgstore.RunDeletion
	*pgstore.ProjectDeletion
	*pgstore.CredentialHolds
}

// NewStore returns every seam on db, a pool or a caller's transaction.
func NewStore(db persistencepostgres.DBTX) *Store {
	return &Store{
		ClaimStore: pgstore.NewClaimStore(db), OwnerStore: pgstore.NewOwnerStore(db),
		ReviewStore: pgstore.NewReviewStore(db), QueryStore: pgstore.NewQueryStore(db),
		FindingEvents: pgstore.NewFindingEvents(db), RunDeletion: pgstore.NewRunDeletion(db),
		ProjectDeletion: pgstore.NewProjectDeletion(db), CredentialHolds: pgstore.NewCredentialHolds(db),
	}
}

// TxStore joins every seam bound to the caller's transaction, including the
// steps that exist only on a seam's transaction-bound store.
type TxStore struct {
	*pgstore.ClaimStore
	*pgstore.OwnerTxStore
	*pgstore.ReviewTxStore
	*pgstore.QueryStore
	*pgstore.FindingEventsTx
	*pgstore.RunDeletionTx
	*pgstore.ProjectDeletion
	*pgstore.CredentialHolds
}

// NewTxStore returns every seam bound to the caller's transaction tx.
func NewTxStore(tx pgx.Tx) *TxStore {
	return &TxStore{
		ClaimStore: pgstore.NewClaimStore(tx), OwnerTxStore: pgstore.NewOwnerTxStore(tx),
		ReviewTxStore: pgstore.NewReviewTxStore(tx), QueryStore: pgstore.NewQueryStore(tx),
		FindingEventsTx: pgstore.NewFindingEventsTx(tx), RunDeletionTx: pgstore.NewRunDeletionTx(tx),
		ProjectDeletion: pgstore.NewProjectDeletion(tx), CredentialHolds: pgstore.NewCredentialHolds(tx),
	}
}

// WithPurgeDependencies composes the actual foreign row owner at the test
// caller, avoiding both a test-helper ownership violation and an import cycle
// in that owner's own tests.
func (s *Store) WithPurgeDependencies(purge auditstore.PurgeDependencies) *Store {
	bound := *s
	bound.ClaimStore = s.ClaimStore.WithPurgeDependencies(purge)
	return &bound
}
