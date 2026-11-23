// Package auditstoretest gives tests one Audit store with every seam, so a
// test can set up and inspect an Audit across the seams the Server reaches
// through several callers. Production code holds only the seam it uses;
// internal/archtest keeps this package out of every non-test package.
package auditstoretest

import (
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/internal/pgstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// Store joins every seam of the Audit store on one database handle.
type Store struct {
	storage *artifactpostgres.Storage
	*pgstore.ClaimStore
	*pgstore.OwnerStore
	*pgstore.QueryStore
	*pgstore.FindingEvents
	*pgstore.RunDeletion
	*pgstore.ProjectDeletion
	*pgstore.CredentialHolds
	db persistencepostgres.DBTX
}

// NewStore returns every seam on db, a pool or a caller's transaction.
func NewStore(db persistencepostgres.DBTX, storage *artifactpostgres.Storage) *Store {
	return &Store{db: db, storage: storage,
		ClaimStore: pgstore.NewClaimStore(db, storage), OwnerStore: pgstore.NewOwnerStore(db, storage),
		QueryStore:    pgstore.NewQueryStore(db, storage),
		FindingEvents: pgstore.NewFindingEvents(db, storage), RunDeletion: pgstore.NewRunDeletion(db, storage),
		ProjectDeletion: pgstore.NewProjectDeletion(db, storage), CredentialHolds: pgstore.NewCredentialHolds(db, storage),
	}
}

// TxStore joins every seam bound to the caller's transaction, including the
// steps that exist only on a seam's transaction-bound store.
type TxStore struct {
	*pgstore.ClaimStore
	*pgstore.OwnerTxStore
	*pgstore.QueryStore
	*pgstore.FindingEventsTx
	*pgstore.RunDeletionTx
	*pgstore.ProjectDeletion
	*pgstore.CredentialHolds
}

// NewTxStore returns every seam bound to the caller's transaction tx.
func NewTxStore(tx pgx.Tx, storage *artifactpostgres.Storage) *TxStore {
	return &TxStore{
		ClaimStore: pgstore.NewClaimStore(tx, storage), OwnerTxStore: pgstore.NewOwnerTxStore(tx, storage),
		QueryStore:      pgstore.NewQueryStore(tx, storage),
		FindingEventsTx: pgstore.NewFindingEventsTx(tx, storage), RunDeletionTx: pgstore.NewRunDeletionTx(tx, storage),
		ProjectDeletion: pgstore.NewProjectDeletion(tx, storage), CredentialHolds: pgstore.NewCredentialHolds(tx, storage),
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
