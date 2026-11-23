package pgstore

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestPostgresWorkspaceKeepsTaskProgressSeparateFromFindings(t *testing.T) {
	f := newCollectingAuditFixture(t, "workspace-progress", 1<<20, collectingAuditFixtureOptions{extraPending: true})
	require := func(total, completed, gaps, pending, findings int) {
		t.Helper()
		got, err := f.store.GetWorkspace(f.ctx, f.audit.OwnerID, f.audit.AuditID)
		if err != nil || got.TotalTasks != total || got.CompletedTasks != completed || got.Gaps != gaps || got.PendingTasks != pending || got.Findings != findings || got.Iteration != 1 || got.AsOf.IsZero() {
			t.Fatalf("workspace = %+v, %v", got, err)
		}
		if got.TotalTasks != got.CompletedTasks+got.Gaps+got.PendingTasks {
			t.Fatal("progress does not partition Tasks")
		}
	}
	require(2, 0, 0, 2, 0)
	// A proposal can be retained even when the Run's ordinary result is missing.
	// The Finding must not turn its Task into completed work or an issue bucket.
	collectPurgeFinding(t, f, "workspace-finding")
	require(2, 0, 1, 1, 1)
	if _, err := f.store.GetWorkspace(f.ctx, "other-owner", f.audit.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign workspace: %v", err)
	}

	success := newCollectingAuditFixture(t, "workspace-completed", 1<<20)
	if _, changed, err := success.store.Collect(success.ctx, storedCheckCollect(success, auditstore.ExactArtifact{})); err != nil || !changed {
		t.Fatalf("collect: %v %v", changed, err)
	}
	got, err := success.store.GetWorkspace(success.ctx, success.audit.OwnerID, success.audit.AuditID)
	if err != nil || got.TotalTasks != 1 || got.CompletedTasks != 1 || got.Gaps != 0 || got.PendingTasks != 0 || got.Findings != 0 {
		t.Fatalf("completed workspace: %+v %v", got, err)
	}
}
