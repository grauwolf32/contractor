package auditstore

import (
	"slices"
	"strings"
	"testing"
)

func TestPostgresSettleUndispatchedPreservesAttemptedCoverage(t *testing.T) {
	for _, test := range []struct {
		name               string
		targetState        AuditState
		attemptedGap       string
		attemptedMessage   string
		unattemptedGap     string
		unattemptedMessage string
		unattemptedState   CoverageStatus
		finalDisposition   FinalDisposition
	}{
		{
			name: "finalizing", targetState: AuditFinalizing,
			attemptedGap:       "audit-closed-before-retry",
			attemptedMessage:   "Audit dispatch closed before a retry of this item was submitted.",
			unattemptedGap:     "audit-closed-before-dispatch",
			unattemptedMessage: "Audit dispatch closed before this item was submitted.",
			unattemptedState:   CoverageNotTested, finalDisposition: FinalExcluded,
		},
		{
			name: "cancelling", targetState: AuditCancelling,
			attemptedGap:       "audit-cancelled-before-retry",
			attemptedMessage:   "Audit cancellation closed this item before a retry was submitted.",
			unattemptedGap:     "audit-cancelled-before-dispatch",
			unattemptedMessage: "Audit cancellation closed this item before dispatch.",
			unattemptedState:   CoverageBlocked, finalDisposition: FinalExecutionCancelled,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			f := newCollectingAuditFixture(t, "settlement-"+test.name, 1<<20,
				collectingAuditFixtureOptions{extraPending: true, terminalOutcome: "failed"})
			const failedRationale = "The child Run failed before an acceptable semantic result was collected."
			if _, inserted, err := f.store.Collect(f.ctx, CollectParams{
				Claim: f.claim, ReceiptID: "receipt-settlement-" + test.name,
				ExecutionID: f.execution.ExecutionID,
				Disposition: CollectionExecutionFailed, ErrorCode: stringPointer("run_failed"),
				RequestDigest: testDigest("5"),
				Items: []CollectionItem{{
					ExecutionItemID: f.memberID, Disposition: CollectionExecutionFailed,
					Retryable: true, FinalDisposition: FinalExecutionFailed,
					Coverage: Coverage{
						Status: CoverageBlocked, Requested: []string{}, Completed: []string{},
						Gaps: []string{"execution-failed"}, Rationale: failedRationale,
					},
				}},
			}); err != nil || !inserted {
				t.Fatalf("collect retryable failure = (%t, %v)", inserted, err)
			}
			items, err := f.store.ListItems(f.ctx, f.audit.AuditID)
			if err != nil || len(items) != 2 || items[0].State != ItemReady ||
				items[0].LastExecutionItemID == nil || items[1].LastExecutionItemID != nil {
				t.Fatalf("items before settlement = (%+v, %v)", items, err)
			}
			audit, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := f.store.TransitionClaimed(f.ctx, ClaimedTransitionParams{
				Claim: f.claim, ExpectedRevision: audit.Revision,
				ExpectedState: AuditActive, TargetState: test.targetState,
			}); err != nil {
				t.Fatal(err)
			}
			settled, err := f.store.SettleUndispatched(f.ctx, f.claim, MaxReconcileRows)
			if err != nil || settled != 2 {
				t.Fatalf("settle items = (%d, %v), want 2", settled, err)
			}
			rows, err := f.store.ListCoverage(f.ctx, f.audit.AuditID, "round-settlement-"+test.name, -1, 10)
			if err != nil || len(rows) != 2 {
				t.Fatalf("settled coverage = (%+v, %v)", rows, err)
			}
			attempted := rows[0].Coverage
			if attempted.Status != CoverageBlocked ||
				!slices.Contains(attempted.Gaps, "execution-failed") ||
				!slices.Contains(attempted.Gaps, test.attemptedGap) ||
				slices.Contains(attempted.Gaps, test.unattemptedGap) ||
				attempted.Rationale != failedRationale+" "+test.attemptedMessage ||
				strings.Contains(attempted.Rationale, "before this item was submitted") {
				t.Errorf("attempted coverage lost failure provenance: %+v", attempted)
			}
			unattempted := rows[1].Coverage
			if unattempted.Status != test.unattemptedState ||
				!slices.Contains(unattempted.Gaps, test.unattemptedGap) ||
				slices.Contains(unattempted.Gaps, test.attemptedGap) ||
				unattempted.Rationale != test.unattemptedMessage {
				t.Errorf("unattempted coverage changed: %+v", unattempted)
			}
			items, err = f.store.ListItems(f.ctx, f.audit.AuditID)
			if err != nil || len(items) != 2 {
				t.Fatalf("settled items = (%+v, %v)", items, err)
			}
			for _, item := range items {
				if item.State != ItemSettled || item.FinalDisposition == nil ||
					*item.FinalDisposition != test.finalDisposition {
					t.Errorf("settled item = %+v", item)
				}
			}
		})
	}
}
