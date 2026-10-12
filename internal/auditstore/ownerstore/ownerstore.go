// Package ownerstore is the Audit store seam of the owner's lifecycle
// commands: draft creation, start (the first Round or preparation), pause,
// cancellation, resume and deletion, each replayed by the owner's
// idempotency key.
package ownerstore

import (
	"github.com/grauwolf32/contractor/internal/auditstore/internal/pgstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// Store runs the owner's lifecycle commands on PostgreSQL.
type Store = pgstore.OwnerStore

// TxStore is a Store bound to the caller's transaction. It adds the owner
// steps whose row locks must last until the caller commits (PER-R18).
type TxStore = pgstore.OwnerTxStore

// New returns the owner's commands on db, a pool or a caller's transaction.
func New(db persistencepostgres.DBTX) *Store {
	return pgstore.NewOwnerStore(db)
}

// NewTx returns the owner's commands bound to the caller's transaction tx.
func NewTx(tx pgx.Tx) *TxStore {
	return pgstore.NewOwnerTxStore(tx)
}
