package auditstoretest

import (
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/internal/pgstore"
)

// ValidateClaimedTransition applies the checks of the claim store's
// TransitionClaimed that need no database, so a test double refuses what the
// store refuses.
func ValidateClaimedTransition(params auditstore.ClaimedTransitionParams) error {
	return pgstore.ValidateClaimedTransition(params)
}

// ValidateCollect applies the checks of the claim store's Collect that need
// no database.
func ValidateCollect(params auditstore.CollectParams) error {
	return pgstore.ValidateCollect(params)
}
