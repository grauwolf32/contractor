package auditimport

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestImporterClosedDispatchUnboundFailureKeepsDrainSemantics(t *testing.T) {
	for _, assessment := range []bool{false, true} {
		t.Run(map[bool]string{false: "check", true: "declared assessment"}[assessment], func(t *testing.T) {
			h := newImportHarness(t)
			if assessment {
				configureDrainAssessment(t, &h)
			}
			runs := failedInputRunReader(&h)
			outcome := auditstore.TerminalSubmissionFailed
			h.execution.TerminalOutcome, h.execution.RunID = &outcome, nil
			h.snapshot.Audit.State, h.snapshot.Audit.Dispatch = auditstore.AuditCancelling, auditstore.DispatchClosed
			changed, err := h.importer.Collect(t.Context(), h.claim, h.snapshot, h.execution)
			got := h.store.collected
			if err != nil || !changed || runs.reads != 0 || got.Disposition != auditstore.CollectionExecutionFailed || got.ErrorCode == nil || *got.ErrorCode != "execution-failed" {
				t.Fatalf("closed-dispatch collection = (%t,%v,%d reads,%+v)", changed, err, runs.reads, got)
			}
			if !assessment && (len(got.Items) != 1 || !got.Items[0].Retryable || got.Items[0].FinalDisposition == auditstore.FinalInvalidResult) {
				t.Fatalf("unbound drain changed member semantics: %+v", got.Items)
			}
		})
	}
}

func TestImporterClosedDispatchKeepsOwnedChildInputFailurePermanent(t *testing.T) {
	h := newImportHarness(t)
	runs := failedInputRunReader(&h)
	h.snapshot.Audit.State, h.snapshot.Audit.Dispatch = auditstore.AuditCancelling, auditstore.DispatchClosed
	changed, err := h.importer.Collect(t.Context(), h.claim, h.snapshot, h.execution)
	got := h.store.collected
	if err != nil || !changed || runs.reads != 1 || got.Disposition != auditstore.CollectionContractInvalid || got.ErrorCode == nil || *got.ErrorCode != runstore.AuditInputInitializationFailedReason || got.Items[0].Retryable {
		t.Fatalf("owned child failure during drain = (%t,%v,%d reads,%+v)", changed, err, runs.reads, got)
	}
}

func TestImporterPermanentSubmissionReceiptReplaysAfterDispatchCloses(t *testing.T) {
	for _, assessment := range []bool{false, true} {
		t.Run(map[bool]string{false: "check", true: "declared assessment"}[assessment], func(t *testing.T) {
			h := newImportHarness(t)
			if assessment {
				configureDrainAssessment(t, &h)
			}
			runs := failedInputRunReader(&h)
			outcome := auditstore.TerminalSubmissionFailed
			h.execution.TerminalOutcome, h.execution.RunID = &outcome, nil
			changed, err := h.importer.Collect(t.Context(), h.claim, h.snapshot, h.execution)
			if err != nil || !changed {
				t.Fatalf("permanent submission collection = (%t,%v)", changed, err)
			}
			first := h.store.collected
			h.snapshot.Audit.State, h.snapshot.Audit.Dispatch = auditstore.AuditCancelling, auditstore.DispatchClosed
			receipt := auditstore.CollectionReceiptSummary{ReceiptID: first.ReceiptID, AuditID: h.execution.AuditID, ExecutionID: h.execution.ExecutionID, TerminalOutcome: outcome, Disposition: first.Disposition, ErrorCode: first.ErrorCode, RequestDigest: first.RequestDigest}
			if assessment {
				h.snapshot.RoleReceipts = []auditstore.CollectionReceiptSummary{receipt}
			} else {
				h.snapshot.Receipts = []auditstore.CollectionReceiptSummary{receipt}
				h.store.selectedItems[0].State = auditstore.ItemSettled
				h.store.members[0].State = auditstore.ItemSettled
			}
			changed, err = h.importer.Collect(t.Context(), h.claim, h.snapshot, h.execution)
			got := h.store.collected
			if err != nil || !changed || runs.reads != 0 || got.Disposition != first.Disposition || got.RequestDigest != first.RequestDigest {
				t.Fatalf("replay after closure = (%t,%v,%d reads,%+v), first %+v", changed, err, runs.reads, got, first)
			}
		})
	}
}

func configureDrainAssessment(t *testing.T, h *importHarness) {
	t.Helper()
	profile := inputFailureAssessmentProfile(t, true)
	data, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	h.snapshot.Audit.ProfileSnapshot = data
	h.snapshot.Audit.Profile = auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest}
	h.execution.Role, h.execution.WorkflowRole = auditstore.ExecutionAssessment, "review"
	attempt := 1
	h.execution.RoleAttempt = &attempt
	h.store.members = nil
	h.snapshot.Round = &auditstore.Round{RoundID: *h.execution.RoundID, Ordinal: 1}
}

func TestImporterPausedOpenDispatchKeepsPermanentSubmissionRejection(t *testing.T) {
	h := newImportHarness(t)
	runs := failedInputRunReader(&h)
	outcome := auditstore.TerminalSubmissionFailed
	h.execution.TerminalOutcome, h.execution.RunID = &outcome, nil
	h.snapshot.Audit.State, h.snapshot.Audit.Dispatch, h.snapshot.Audit.Hold = auditstore.AuditPaused, auditstore.DispatchOpen, auditstore.HoldHeld
	changed, err := h.importer.Collect(t.Context(), h.claim, h.snapshot, h.execution)
	got := h.store.collected
	if err != nil || !changed || runs.reads != 0 || got.Disposition != auditstore.CollectionContractInvalid || got.Items[0].Retryable {
		t.Fatalf("paused permanent rejection = (%t,%v,%d reads,%+v)", changed, err, runs.reads, got)
	}
}

func TestImporterClosedDispatchIgnoresUnrelatedPermanentReceipt(t *testing.T) {
	for name, mutate := range map[string]func(*auditstore.CollectionReceiptSummary){
		"Audit":       func(r *auditstore.CollectionReceiptSummary) { r.AuditID = "another" },
		"execution":   func(r *auditstore.CollectionReceiptSummary) { r.ExecutionID = "another" },
		"outcome":     func(r *auditstore.CollectionReceiptSummary) { r.TerminalOutcome = auditstore.TerminalFailed },
		"Run":         func(r *auditstore.CollectionReceiptSummary) { r.RunID = stringPointer("another") },
		"generation":  func(r *auditstore.CollectionReceiptSummary) { r.TerminalRunGeneration = stringPointer("another") },
		"sequence":    func(r *auditstore.CollectionReceiptSummary) { n := uint64(1); r.TerminalRunSequence = &n },
		"disposition": func(r *auditstore.CollectionReceiptSummary) { r.Disposition = auditstore.CollectionExecutionFailed },
		"code":        func(r *auditstore.CollectionReceiptSummary) { r.ErrorCode = stringPointer("another") },
	} {
		t.Run(name, func(t *testing.T) {
			h := newImportHarness(t)
			configureDrainAssessment(t, &h)
			runs := failedInputRunReader(&h)
			outcome := auditstore.TerminalSubmissionFailed
			h.execution.TerminalOutcome, h.execution.RunID = &outcome, nil
			h.snapshot.Audit.State, h.snapshot.Audit.Dispatch = auditstore.AuditCancelling, auditstore.DispatchClosed
			receipt := auditstore.CollectionReceiptSummary{AuditID: h.execution.AuditID, ExecutionID: h.execution.ExecutionID, TerminalOutcome: outcome, Disposition: auditstore.CollectionContractInvalid, ErrorCode: stringPointer(runstore.AuditInputInitializationFailedReason)}
			mutate(&receipt)
			h.snapshot.RoleReceipts = []auditstore.CollectionReceiptSummary{receipt}
			changed, err := h.importer.Collect(t.Context(), h.claim, h.snapshot, h.execution)
			if err != nil || !changed || runs.reads != 0 || h.store.collected.Disposition != auditstore.CollectionExecutionFailed {
				t.Fatalf("unrelated receipt changed drain = (%t,%v,%+v)", changed, err, h.store.collected)
			}
		})
	}
}
