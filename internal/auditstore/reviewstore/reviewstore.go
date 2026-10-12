// Package reviewstore is the Audit store seam of the review journal: finding
// triage requests and decisions, item action decisions, report candidate
// decisions and the review events they record.
package reviewstore

import (
	"github.com/grauwolf32/contractor/internal/auditstore/internal/pgstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// Store runs the review journal's writes on PostgreSQL.
type Store = pgstore.ReviewStore

// TxStore is a Store bound to the caller's transaction. It adds the review
// steps whose row locks must last until the caller commits (PER-R18).
type TxStore = pgstore.ReviewTxStore

// New returns the review journal on db, a pool or a caller's transaction.
func New(db persistencepostgres.DBTX) *Store {
	return pgstore.NewReviewStore(db)
}

// NewTx returns the review journal bound to the caller's transaction tx.
func NewTx(tx pgx.Tx) *TxStore {
	return pgstore.NewReviewTxStore(tx)
}
