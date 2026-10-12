package auditcontroller

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// claimedTransitionRefusal is one claimed transition that both
// fakeControllerStore and claimstore.Store must refuse, applied to an
// active Audit at revision with a live claim (V368-010).
type claimedTransitionRefusal struct {
	name   string
	params func(claim auditstore.ControllerClaim, revision uint64) auditstore.ClaimedTransitionParams
	want   error
}

func claimedTransitionRefusals() []claimedTransitionRefusal {
	transition := func(from, to auditstore.AuditState) func(auditstore.ControllerClaim, uint64) auditstore.ClaimedTransitionParams {
		return func(claim auditstore.ControllerClaim, revision uint64) auditstore.ClaimedTransitionParams {
			return auditstore.ClaimedTransitionParams{
				Claim: claim, ExpectedRevision: revision, ExpectedState: from, TargetState: to,
			}
		}
	}
	return []claimedTransitionRefusal{
		{"active to completed skips finalizing", transition(auditstore.AuditActive, auditstore.AuditCompleted), auditstore.ErrInvalid},
		{"active to cancelled skips cancelling", transition(auditstore.AuditActive, auditstore.AuditCancelled), auditstore.ErrInvalid},
		{"active back to draft", transition(auditstore.AuditActive, auditstore.AuditDraft), auditstore.ErrInvalid},
		{"active to active", transition(auditstore.AuditActive, auditstore.AuditActive), auditstore.ErrInvalid},
		{"draft to active belongs to preparation", transition(auditstore.AuditDraft, auditstore.AuditActive), auditstore.ErrInvalid},
		{"unknown target state", transition(auditstore.AuditActive, "archived"), auditstore.ErrInvalid},
		{"no expected revision", func(claim auditstore.ControllerClaim, _ uint64) auditstore.ClaimedTransitionParams {
			return transition(auditstore.AuditActive, auditstore.AuditPaused)(claim, 0)
		}, auditstore.ErrInvalid},
		{"reason without a code", func(claim auditstore.ControllerClaim, revision uint64) auditstore.ClaimedTransitionParams {
			params := transition(auditstore.AuditActive, auditstore.AuditPaused)(claim, revision)
			params.Reason = &auditstore.StopReason{Code: " ", Message: "no code"}
			return params
		}, auditstore.ErrInvalid},
		{"claim of a later epoch", func(claim auditstore.ControllerClaim, revision uint64) auditstore.ClaimedTransitionParams {
			claim.Epoch++
			return transition(auditstore.AuditActive, auditstore.AuditPaused)(claim, revision)
		}, auditstore.ErrClaimLost},
		{"stale revision", func(claim auditstore.ControllerClaim, revision uint64) auditstore.ClaimedTransitionParams {
			return transition(auditstore.AuditActive, auditstore.AuditPaused)(claim, revision+5)
		}, auditstore.ErrPrecondition},
	}
}

// refuseClaimedTransitions applies every refusal to an active Audit at
// revision through store and reports how each one failed. None may change
// the Audit; the caller checks that through its own store.
func refuseClaimedTransitions(
	ctx context.Context,
	store interface {
		TransitionClaimed(context.Context, auditstore.ClaimedTransitionParams) (auditstore.Audit, error)
	},
	claim auditstore.ControllerClaim,
	revision uint64,
) []error {
	results := make([]error, 0, len(claimedTransitionRefusals()))
	for _, refusal := range claimedTransitionRefusals() {
		_, err := store.TransitionClaimed(ctx, refusal.params(claim, revision))
		results = append(results, err)
	}
	return results
}

func checkClaimedTransitionRefusals(t *testing.T, name string, results []error) {
	t.Helper()
	for index, refusal := range claimedTransitionRefusals() {
		if !errors.Is(results[index], refusal.want) {
			t.Errorf("%s %s = %v, want %v", name, refusal.name, results[index], refusal.want)
		}
	}
}

// TestFakeControllerStoreRefusesInvalidClaimedTransitions holds the fake to
// the store's transition table and claim fence;
// TestPostgresStoreRefusesTheSameClaimedTransitions drives the store itself.
func TestFakeControllerStoreRefusesInvalidClaimedTransitions(t *testing.T) {
	ctx := t.Context()
	store := newFakeControllerStore(t, 1, 1)
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "parity", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim = (%+v, %v)", claims, err)
	}
	revision := store.audit.Revision
	checkClaimedTransitionRefusals(t, "fake", refuseClaimedTransitions(ctx, store, claims[0], revision))
	if store.audit.State != auditstore.AuditActive || store.audit.Revision != revision {
		t.Fatalf("refused transitions changed the Audit to %s at revision %d", store.audit.State, store.audit.Revision)
	}
}
