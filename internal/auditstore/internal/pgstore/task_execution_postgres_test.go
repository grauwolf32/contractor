package pgstore

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/jackc/pgx/v5"
)

func TestSharedPostgresCommonTaskSelectionReservesOneStableAttempt(t *testing.T) {
	f := newCommonTaskFixture(t, 1)
	ctx := t.Context()
	accepted, _, err := f.owners.AcceptTasks(ctx, f.acceptance(2, "accept"))
	if err != nil {
		t.Fatal(err)
	}
	claims := NewClaimStore(f.pool)
	claimed, err := claims.Claim(ctx, auditstore.ClaimParams{HolderID: "task-selector", Lease: time.Minute, Limit: 1})
	if err != nil || len(claimed) != 1 {
		t.Fatalf("claim: %+v %v", claimed, err)
	}
	params := auditstore.SelectTasksParams{Claim: claimed[0], TaskIDs: accepted.TaskIDs[:1]}
	execution, created, err := claims.SelectTasks(ctx, params)
	if err != nil || !created {
		t.Fatalf("select: %+v %v %v", execution, created, err)
	}
	if execution.Role != auditstore.ExecutionTask || execution.RoundID != nil || execution.Manifest.Ref.Revision != nil || execution.RunID != nil || execution.State != auditstore.ExecutionIntent {
		t.Fatalf("execution contains old authoring authority: %+v", execution)
	}
	replay, created, err := claims.SelectTasks(ctx, params)
	if err != nil || created || replay.ExecutionID != execution.ExecutionID || replay.SubmissionKey != execution.SubmissionKey {
		t.Fatalf("reservation replay: %+v %v %v", replay, created, err)
	}
	task, err := f.reads.GetTask(ctx, "owner", f.audit.AuditID, accepted.TaskIDs[0])
	if err != nil || task.State != auditstore.TaskSelected || task.Attempts != 1 || task.LastExecution == nil || *task.LastExecution != execution.ExecutionID {
		t.Fatalf("selected task: %+v %v", task, err)
	}
	audit, err := f.reads.Get(ctx, "owner", f.audit.AuditID)
	if err != nil || audit.ReservedRunCount != 1 || audit.OutstandingRunCount != 1 || audit.SubmittedRunCount != 0 {
		t.Fatalf("reserved counters: %+v %v", audit, err)
	}
	paused, _, err := f.owners.Transition(ctx, auditstore.TransitionParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused, IdempotencyKey: "pause", RequestDigest: contentdigest.Bytes([]byte("pause"))})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := claims.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claimed[0], TaskIDs: accepted.TaskIDs[1:]}); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("paused selection: %v", err)
	}
	if replay, created, err := claims.SelectTasks(ctx, params); err != nil || created || replay.ExecutionID != execution.ExecutionID {
		t.Fatalf("paused original intent reconciliation: %+v %v %v", replay, created, err)
	}
	after, err := f.reads.Get(ctx, "owner", f.audit.AuditID)
	if err != nil || after.Revision != paused.Revision || after.ReservedRunCount != 1 {
		t.Fatalf("refused selection mutated Audit: %+v %v", after, err)
	}
}

func TestSharedPostgresCommonTaskCollectionAndPurge(t *testing.T) {
	f := newCommonTaskFixture(t, 1)
	ctx := t.Context()
	accepted, _, err := f.owners.AcceptTasks(ctx, f.acceptance(1, "accept"))
	if err != nil {
		t.Fatal(err)
	}
	claims := NewClaimStore(f.pool).WithPurgeDependencies(func(context.Context, pgx.Tx, string, int) (int, error) { return 0, nil })
	claimed, err := claims.Claim(ctx, auditstore.ClaimParams{HolderID: "common-purger", Lease: time.Minute, Limit: 1})
	if err != nil || len(claimed) != 1 {
		t.Fatalf("claim: %+v %v", claimed, err)
	}
	claim := claimed[0]
	execution, _, err := claims.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claim, TaskIDs: accepted.TaskIDs})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := claims.ObserveSubmissionFailure(ctx, auditstore.ObserveSubmissionFailureParams{Claim: claim, ExecutionID: execution.ExecutionID}); err != nil {
		t.Fatal(err)
	}
	gap := "submission-failed"
	params := auditstore.CollectParams{Claim: claim, ExecutionID: execution.ExecutionID, ReceiptID: "common-receipt", Disposition: auditstore.CollectionExecutionFailed, RequestDigest: contentdigest.Bytes([]byte("collect")), Retained: []auditstore.ArtifactLink{}, Tasks: []auditstore.TaskCollection{{ExecutionItemID: auditdomain.DeterministicID("attempt", execution.ExecutionID, accepted.TaskIDs[0]), TaskID: accepted.TaskIDs[0], GapCode: &gap}}}
	first, changed, err := claims.Collect(ctx, params)
	if err != nil || !changed {
		t.Fatalf("collect: %+v %v %v", first, changed, err)
	}
	replayed, changed, err := claims.Collect(ctx, params)
	if err != nil || changed || replayed.ReceiptID != first.ReceiptID {
		t.Fatalf("replay: %+v %v %v", replayed, changed, err)
	}
	task, err := f.reads.GetTask(ctx, "owner", f.audit.AuditID, accepted.TaskIDs[0])
	if err != nil || task.State != auditstore.TaskSettled || task.GapCode == nil || *task.GapCode != gap || task.LastExecution == nil || *task.LastExecution != execution.ExecutionID {
		t.Fatalf("collected Task: %+v %v", task, err)
	}
	var memberCount int
	if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM audit_execution_items WHERE audit_id=$1 AND task_id=$2`, f.audit.AuditID, task.Document.TaskID).Scan(&memberCount); err != nil || memberCount != 1 {
		t.Fatalf("Task attempt FK unexercised: %d %v", memberCount, err)
	}
	audit, err := f.reads.Get(ctx, "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	audit, _, err = f.owners.RequestDelete(ctx, auditstore.DeleteParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "delete", RequestDigest: contentdigest.Bytes([]byte("delete"))})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := claims.ReleaseDispatchHold(ctx, claim); err != nil {
		t.Fatal(err)
	}
	audit, err = f.reads.Get(ctx, "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := claims.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: audit.Revision, ExpectedState: auditstore.AuditCancelling, TargetState: auditstore.AuditDeleting}); err != nil {
		t.Fatal(err)
	}
	for range 10 {
		if err := claims.PurgeClaimed(ctx, claim, auditdomain.ArtifactNamespace(audit.AuditID)); err != nil {
			t.Fatal(err)
		}
		var exists bool
		if err := f.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audits WHERE audit_id=$1)`, audit.AuditID).Scan(&exists); err != nil {
			t.Fatal(err)
		}
		if !exists {
			break
		}
	}
	for _, table := range []string{"audits", "audit_tasks", "audit_execution_items", "audit_executions"} {
		var count int
		if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM `+pgx.Identifier{table}.Sanitize()+` WHERE audit_id=$1`, audit.AuditID).Scan(&count); err != nil || count != 0 {
			t.Fatalf("%s retained %d rows: %v", table, count, err)
		}
	}
}

func TestSharedPostgresCommonTaskSelectionRejectsUnattributableBatch(t *testing.T) {
	f := newCommonTaskFixture(t, 1)
	accepted, _, err := f.owners.AcceptTasks(t.Context(), f.acceptance(2, "accept"))
	if err != nil {
		t.Fatal(err)
	}
	claims := NewClaimStore(f.pool)
	claimed, err := claims.Claim(t.Context(), auditstore.ClaimParams{HolderID: "selector", Lease: time.Minute, Limit: 1})
	if err != nil || len(claimed) != 1 {
		t.Fatalf("claim: %+v %v", claimed, err)
	}
	// This Workflow has one ordinary selected source and no per-item context.
	// It cannot truthfully distinguish two tasks even if the bytes coincide.
	if _, _, err := claims.SelectTasks(t.Context(), auditstore.SelectTasksParams{Claim: claimed[0], TaskIDs: accepted.TaskIDs}); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("batch without context: %v", err)
	}
	var count int
	if err := f.pool.QueryRow(t.Context(), `SELECT count(*) FROM audit_executions WHERE audit_id=$1`, f.audit.AuditID).Scan(&count); err != nil || count != 0 {
		t.Fatalf("partial reservation: %d %v", count, err)
	}
}
