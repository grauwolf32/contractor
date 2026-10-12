// Package querystore is the Audit store seam of the reads: Audits, Rounds,
// items and their attempts, executions, coverage, artifact links,
// preparation and the pages of a report build.
package querystore

import (
	"github.com/grauwolf32/contractor/internal/auditstore/internal/pgstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// Store runs the Audit reads on PostgreSQL.
type Store = pgstore.QueryStore

// New returns the Audit reads on db, a pool or a caller's transaction.
func New(db persistencepostgres.DBTX) *Store {
	return pgstore.NewQueryStore(db)
}
