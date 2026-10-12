package pgstore

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// TestOwnerTransitionsOnlyPauseOrCancel pins the owner transition table that
// Transition's SQL relies on: an owner only pauses or cancels. Resume alone
// reopens dispatch behind the active-Project gate, and every other target is
// Controller authority.
func TestOwnerTransitionsOnlyPauseOrCancel(t *testing.T) {
	t.Parallel()
	states := []auditstore.AuditState{
		auditstore.AuditDraft, auditstore.AuditActive, auditstore.AuditWaitingReview, auditstore.AuditPaused, auditstore.AuditFinalizing,
		auditstore.AuditCancelling, auditstore.AuditCompleted, auditstore.AuditCancelled, auditstore.AuditFailed, auditstore.AuditDeleting,
	}
	allowed := 0
	for _, from := range states {
		for _, to := range states {
			err := validateTransition(auditstore.TransitionParams{
				OwnerID: "owner", AuditID: "audit", ExpectedRevision: 2,
				ExpectedState: from, TargetState: to,
				IdempotencyKey: "transition", RequestDigest: testDigest("1"),
			})
			if err != nil {
				continue
			}
			allowed++
			if to != auditstore.AuditPaused && to != auditstore.AuditCancelling {
				t.Errorf("owner transition %s -> %s is accepted", from, to)
			}
		}
	}
	if allowed != 6 {
		t.Fatalf("owner transitions = %d, want 2 pauses and 4 cancellations", allowed)
	}
}
