// Package claimstore is the Audit store seam of the commands the Audit
// controller runs under a live controller claim: claims, claimed transitions,
// Round acceptance, execution intents, child Run binding, terminal
// observation, collection, report finalization, settlement and purge. Only
// the controller side imports it; the owner-facing Audit service must not
// (internal/archtest), so no owner request can act with a controller's
// authority.
package claimstore

import (
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/internal/pgstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// Store runs the claim-bound commands on PostgreSQL. Every command checks the
// claim's holder and epoch in the statement that applies it (AUX-R74).
type Store = pgstore.ClaimStore

// New returns the claim-bound commands on db, a pool or a caller's
// transaction.
func New(db persistencepostgres.DBTX) *Store {
	return pgstore.NewClaimStore(db)
}

// ValidateClaimedTransition applies the checks of Store.TransitionClaimed
// that need no database, so a test double refuses what the store refuses.
func ValidateClaimedTransition(params auditstore.ClaimedTransitionParams) error {
	return pgstore.ValidateClaimedTransition(params)
}

// ValidateCollect applies the checks of Store.Collect that need no database.
func ValidateCollect(params auditstore.CollectParams) error {
	return pgstore.ValidateCollect(params)
}

// ValidateRecordReportPage applies the page contract of
// Store.RecordReportPage that needs no database.
func ValidateRecordReportPage(params auditstore.RecordReportPageParams) error {
	return pgstore.ValidateRecordReportPage(params)
}
