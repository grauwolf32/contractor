//go:build integration

package auditimport

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestImporterPostgresSettlesPermanentInputFailureOnce(t *testing.T) {
	pool := completionPool(t)
	for _, child := range []bool{false, true} {
		t.Run(map[bool]string{false: "first slice rejected", true: "bound child failed"}[child], func(t *testing.T) {
			f := newCompletionFixtureWithOptions(t, pool, completionOptions{acceptance: "automatic", skipInitialRun: !child})
			var execution auditstore.Execution
			wantCode := "collection-contract-invalid"
			if child {
				mustCompletion(t, artifactstest.FreezeOutputs(f.ctx, f.artifacts, f.run.RunID))
				_, err := f.runs.TransitionRun(f.ctx, f.run.RunID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: runstore.AuditInputInitializationFailedReason})
				mustCompletion(t, err)
				cursor, err := f.runs.GetRunEventCursor(f.ctx, f.run.RunID)
				mustCompletion(t, err)
				execution, err = f.audits.ObserveTerminal(f.ctx, auditstore.ObserveTerminalParams{Claim: f.claim, ExecutionID: f.execution.ExecutionID, RunID: f.run.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)})
				mustCompletion(t, err)
				wantCode = runstore.AuditInputInitializationFailedReason
			} else {
				id := f.id + "-rejected"
				var members []auditstore.ExecutionMemberIntent
				for i, task := range f.tasks {
					members = append(members, auditstore.ExecutionMemberIntent{ExecutionItemID: id + "-" + f.keys[i], ItemID: f.id + "-" + f.keys[i], BatchOrdinal: i, ItemAttempt: 1, Task: task, Inputs: []auditstore.ExactArtifact{f.source, f.taskSet}})
				}
				_, _, err := f.audits.CreateExecutionIntent(f.ctx, auditstore.CreateExecutionIntentParams{Claim: f.claim, ExecutionID: id, RoundID: &f.id, Role: auditstore.ExecutionCheck, WorkflowRole: "check", LaneClass: resolved.LaneClassModel, Manifest: f.manifest, SubmissionKey: id, RequestDigest: completionDigest([]byte(id)), Members: members})
				mustCompletion(t, err)
				execution, err = f.audits.ObserveSubmissionFailure(f.ctx, auditstore.ObserveSubmissionFailureParams{Claim: f.claim, ExecutionID: id})
				mustCompletion(t, err)
			}
			snapshot, err := f.audits.GetReconcileSnapshot(f.ctx, f.claim)
			mustCompletion(t, err)
			access, err := NewArtifactAccess(f.artifacts, hookstore.NewClaimedArtifacts(pool))
			mustCompletion(t, err)
			importer, err := New(f.audits, f.audits, f.runs, access)
			mustCompletion(t, err)
			changed, err := importer.Collect(f.ctx, f.claim, snapshot, execution)
			mustCompletion(t, err)
			if !changed {
				t.Fatal("permanent input failure did not settle")
			}
			changed, err = importer.Collect(f.ctx, f.claim, snapshot, execution)
			mustCompletion(t, err)
			if !changed {
				t.Fatal("replayed collection was not acknowledged")
			}
			settled, err := f.audits.GetReconcileSnapshot(f.ctx, f.claim)
			mustCompletion(t, err)
			wantSubmitted := 0
			if child {
				wantSubmitted = 1
			}
			if len(settled.ReadyItems) != 0 || settled.Audit.SubmittedRunCount != wantSubmitted || len(settled.Receipts) != 1 {
				t.Fatalf("permanent failure readmitted work: ready=%d submitted=%d receipts=%d", len(settled.ReadyItems), settled.Audit.SubmittedRunCount, len(settled.Receipts))
			}
			receipt := settled.Receipts[0]
			if receipt.ExecutionID != execution.ExecutionID || receipt.Disposition != auditstore.CollectionContractInvalid || receipt.ErrorCode == nil || *receipt.ErrorCode != wantCode || (receipt.RunID != nil) != child {
				t.Fatalf("permanent failure receipt = %+v", receipt)
			}
			items, err := f.audits.ListItemsByIDs(f.ctx, f.id, []string{f.id + "-" + f.keys[0], f.id + "-" + f.keys[1]})
			mustCompletion(t, err)
			for _, item := range items {
				if item.State != auditstore.ItemSettled || item.FinalDisposition == nil || *item.FinalDisposition != auditstore.FinalInvalidResult || item.LastExecutionItemID == nil || item.AcceptedResult != nil {
					t.Fatalf("permanent failure item = %+v", item)
				}
			}
		})
	}
}
