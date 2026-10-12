package pgstore

import (
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestPostgresClaimsPausedAuditOnlyWithInFlightExecutions(t *testing.T) {
	f := newCollectingAuditFixture(t, "paused", 1<<20)
	paused, _, err := f.store.Transition(f.ctx, auditstore.TransitionParams{
		OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID,
		ExpectedRevision: f.audit.Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused,
		IdempotencyKey: "pause", RequestDigest: testDigest("a"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.store.ReleaseClaim(f.ctx, f.claim); err != nil {
		t.Fatal(err)
	}
	claim := claimOne(t, f, "collecting")
	if claim == nil {
		t.Fatal("paused Audit with a collecting execution was not claimed")
	}
	code := "collection-contract-invalid"
	if _, _, err := f.store.Collect(f.ctx, auditstore.CollectParams{
		Claim: *claim, ReceiptID: "receipt-paused", ExecutionID: f.execution.ExecutionID,
		Disposition: auditstore.CollectionContractInvalid, ErrorCode: &code, RequestDigest: testDigest("b"),
		Items: []auditstore.CollectionItem{{
			ExecutionItemID: f.memberID, Disposition: auditstore.CollectionContractInvalid,
			FinalDisposition: auditstore.FinalInvalidResult,
			Coverage:         auditstore.Coverage{Status: auditstore.CoverageBlocked, Requested: []string{}, Completed: []string{}, Gaps: []string{code}},
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := f.store.ReleaseClaim(f.ctx, *claim); err != nil {
		t.Fatal(err)
	}
	if idle := claimOne(t, f, "idle"); idle != nil {
		t.Fatalf("idle paused Audit was claimed: %+v", idle)
	}
	paused, err = f.store.Get(f.ctx, paused.OwnerID, paused.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := f.store.Transition(f.ctx, auditstore.TransitionParams{
		OwnerID: paused.OwnerID, AuditID: paused.AuditID,
		ExpectedRevision: paused.Revision, ExpectedState: auditstore.AuditPaused, TargetState: auditstore.AuditCancelling,
		Reason:         &auditstore.StopReason{Code: "cancel_requested", Message: "cancelled by test"},
		IdempotencyKey: "cancel", RequestDigest: testDigest("c"),
	}); err != nil {
		t.Fatal(err)
	}
	if cancelling := claimOne(t, f, "cancelling"); cancelling == nil {
		t.Fatal("cancelling Audit that was paused is not claimable")
	}
}

func claimOne(t *testing.T, f collectingAuditFixture, holder string) *auditstore.ControllerClaim {
	t.Helper()
	claims, err := f.store.Claim(f.ctx, auditstore.ClaimParams{HolderID: holder, Lease: 30 * time.Second, Limit: 1})
	if err != nil || len(claims) > 1 {
		t.Fatalf("claim = (%+v, %v)", claims, err)
	}
	if len(claims) == 0 {
		return nil
	}
	return &claims[0]
}
