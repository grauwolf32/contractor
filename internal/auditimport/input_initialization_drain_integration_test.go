//go:build integration

package auditimport

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestImporterPostgresClosedDispatchPreservesSubmissionDrain(t *testing.T) {
	f := newCompletionFixtureWithOptions(t, completionPool(t), completionOptions{acceptance: "automatic", skipInitialRun: true})
	execution := createUnboundDrainIntent(t, f)
	cancelInputFailureAudit(t, f)
	execution, err := f.audits.ObserveSubmissionFailure(f.ctx, auditstore.ObserveSubmissionFailureParams{Claim: f.claim, ExecutionID: execution.ExecutionID})
	mustCompletion(t, err)
	snapshot, err := f.audits.GetReconcileSnapshot(f.ctx, f.claim)
	mustCompletion(t, err)
	changed, err := drainImporter(t, f).Collect(f.ctx, f.claim, snapshot, execution)
	mustCompletion(t, err)
	if !changed {
		t.Fatal("closed-dispatch intent was not collected")
	}
	settled, err := f.audits.SettleUndispatched(f.ctx, f.claim, auditstore.MaxReconcileRows)
	mustCompletion(t, err)
	if settled != len(f.keys) {
		t.Fatalf("drain settled %d items, want %d ordinary cancelled members", settled, len(f.keys))
	}
	assertInputDrainReceipt(t, f, execution.ExecutionID, auditstore.CollectionExecutionFailed, "execution-failed", auditstore.FinalExecutionCancelled)
}

func TestImporterPostgresPermanentInputReceiptReplaysAfterCancellation(t *testing.T) {
	pool := completionPool(t)
	for _, child := range []bool{false, true} {
		t.Run(map[bool]string{false: "unbound rejection", true: "owned child"}[child], func(t *testing.T) {
			f := newCompletionFixtureWithOptions(t, pool, completionOptions{acceptance: "automatic", skipInitialRun: !child})
			var execution auditstore.Execution
			wantCode := "collection-contract-invalid"
			if child {
				_, err := f.runs.TransitionRun(f.ctx, f.run.RunID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: runstore.AuditInputInitializationFailedReason})
				mustCompletion(t, err)
				cursor, err := f.runs.GetRunEventCursor(f.ctx, f.run.RunID)
				mustCompletion(t, err)
				execution, err = f.audits.ObserveTerminal(f.ctx, auditstore.ObserveTerminalParams{Claim: f.claim, ExecutionID: f.execution.ExecutionID, RunID: f.run.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)})
				mustCompletion(t, err)
				wantCode = runstore.AuditInputInitializationFailedReason
			} else {
				execution = createUnboundDrainIntent(t, f)
				var err error
				execution, err = f.audits.ObserveSubmissionFailure(f.ctx, auditstore.ObserveSubmissionFailureParams{Claim: f.claim, ExecutionID: execution.ExecutionID})
				mustCompletion(t, err)
			}
			importer := drainImporter(t, f)
			snapshot, err := f.audits.GetReconcileSnapshot(f.ctx, f.claim)
			mustCompletion(t, err)
			changed, err := importer.Collect(f.ctx, f.claim, snapshot, execution)
			mustCompletion(t, err)
			if !changed {
				t.Fatal("permanent input rejection did not commit")
			}
			before := assertInputDrainReceipt(t, f, execution.ExecutionID, auditstore.CollectionContractInvalid, wantCode, auditstore.FinalInvalidResult)
			cancelInputFailureAudit(t, f)
			snapshot, err = f.audits.GetReconcileSnapshot(f.ctx, f.claim)
			mustCompletion(t, err)
			changed, err = importer.Collect(f.ctx, f.claim, snapshot, execution)
			mustCompletion(t, err)
			if !changed {
				t.Fatal("lost-response replay was not acknowledged")
			}
			after := assertInputDrainReceipt(t, f, execution.ExecutionID, auditstore.CollectionContractInvalid, wantCode, auditstore.FinalInvalidResult)
			if after.RequestDigest != before.RequestDigest || after.ReceiptID != before.ReceiptID {
				t.Fatalf("closure changed permanent receipt: before %+v; after %+v", before, after)
			}
			settled, err := f.audits.SettleUndispatched(f.ctx, f.claim, auditstore.MaxReconcileRows)
			mustCompletion(t, err)
			if settled != 0 {
				t.Fatalf("replay reopened %d permanent members", settled)
			}
		})
	}
}

func createUnboundDrainIntent(t *testing.T, f *completionFixture) auditstore.Execution {
	t.Helper()
	id := f.id + "-unbound"
	members := make([]auditstore.ExecutionMemberIntent, len(f.tasks))
	for i, task := range f.tasks {
		members[i] = auditstore.ExecutionMemberIntent{ExecutionItemID: id + "-" + f.keys[i], ItemID: f.id + "-" + f.keys[i], BatchOrdinal: i, ItemAttempt: 1, Task: task, Inputs: []auditstore.ExactArtifact{f.source, f.taskSet}}
	}
	execution, _, err := f.audits.CreateExecutionIntent(f.ctx, auditstore.CreateExecutionIntentParams{Claim: f.claim, ExecutionID: id, RoundID: &f.id, Role: auditstore.ExecutionCheck, WorkflowRole: "check", LaneClass: resolved.LaneClassModel, Manifest: f.manifest, SubmissionKey: id, RequestDigest: completionDigest([]byte(id)), Members: members})
	mustCompletion(t, err)
	return execution
}

func cancelInputFailureAudit(t *testing.T, f *completionFixture) {
	t.Helper()
	current, err := f.audits.Get(f.ctx, "owner", f.id)
	mustCompletion(t, err)
	_, _, err = f.audits.Transition(f.ctx, auditstore.TransitionParams{OwnerID: "owner", AuditID: f.id, ExpectedRevision: current.Revision, ExpectedState: current.State, TargetState: auditstore.AuditCancelling, IdempotencyKey: "cancel-" + f.id, RequestDigest: completionDigest([]byte("cancel"))})
	mustCompletion(t, err)
}

func drainImporter(t *testing.T, f *completionFixture) *Importer {
	t.Helper()
	access, err := NewArtifactAccess(f.artifacts, hookstore.NewClaimedArtifacts(f.pool, artifactpgtest.NewStorage()))
	mustCompletion(t, err)
	importer, err := New(f.audits, f.audits, f.runs, access)
	mustCompletion(t, err)
	return importer
}

func assertInputDrainReceipt(t *testing.T, f *completionFixture, executionID string, disposition auditstore.CollectionDisposition, code string, final auditstore.FinalDisposition) auditstore.CollectionReceiptSummary {
	t.Helper()
	snapshot, err := f.audits.GetReconcileSnapshot(f.ctx, f.claim)
	mustCompletion(t, err)
	if len(snapshot.Receipts) != 1 || len(snapshot.ReadyItems) != 0 || snapshot.Audit.OutstandingRunCount != 0 {
		t.Fatalf("drain receipt reopened work: receipts=%d ready=%d outstanding=%d", len(snapshot.Receipts), len(snapshot.ReadyItems), snapshot.Audit.OutstandingRunCount)
	}
	receipt := snapshot.Receipts[0]
	if receipt.ExecutionID != executionID || receipt.Disposition != disposition || receipt.ErrorCode == nil || *receipt.ErrorCode != code {
		t.Fatalf("drain receipt = %+v", receipt)
	}
	ids := make([]string, len(f.keys))
	for i, key := range f.keys {
		ids[i] = f.id + "-" + key
	}
	items, err := f.audits.ListItemsByIDs(f.ctx, f.id, ids)
	mustCompletion(t, err)
	for _, item := range items {
		if item.State != auditstore.ItemSettled || item.FinalDisposition == nil || *item.FinalDisposition != final || item.AcceptedResult != nil {
			t.Fatalf("drain member = %+v, want %s", item, final)
		}
	}
	var runs, receipts int
	mustCompletion(t, f.pool.QueryRow(f.ctx, `SELECT count(run_id),count(collection_receipt_id) FROM audit_executions WHERE audit_id=$1`, f.id).Scan(&runs, &receipts))
	if runs != snapshot.Audit.SubmittedRunCount || receipts != 1 {
		t.Fatalf("drain duplicated charge or receipt: Runs=%d submitted=%d receipts=%d", runs, snapshot.Audit.SubmittedRunCount, receipts)
	}
	return receipt
}
