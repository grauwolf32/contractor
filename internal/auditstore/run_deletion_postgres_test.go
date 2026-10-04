package auditstore

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresDeletedRunInvalidationPreservesFinalizingReportTime(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-run-invalidation", OwnerID: "owner-run-invalidation",
		Kind: projectstore.KindProject, Name: "Run invalidation",
		IdempotencyKey: "create-project", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := NewPostgresStore(pool)
	for _, auditID := range []string{"audit-finalizing", "audit-draft"} {
		if _, _, err := store.CreateDraft(ctx, CreateDraftParams{
			AuditID: auditID, OwnerID: project.OwnerID, ProjectID: project.ProjectID,
			Profile:         ProfileIdentity{Name: "run-invalidation", Version: "1", Digest: testDigest("2")},
			ProfileSnapshot: json.RawMessage(`{}`), InputSelection: json.RawMessage(`{}`),
			Limits: Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
				MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024},
			IdempotencyKey: "create-" + auditID, RequestDigest: testDigest("3"),
		}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := pool.Exec(ctx, `
UPDATE audits
   SET state='finalizing', baseline_snapshot='{}'::jsonb,
       started_at=clock_timestamp(), deadline_at=clock_timestamp()+interval '1 hour'
 WHERE audit_id='audit-finalizing'`); err != nil {
		t.Fatal(err)
	}
	finalizingBefore, err := store.Get(ctx, project.OwnerID, "audit-finalizing")
	if err != nil {
		t.Fatal(err)
	}
	draftBefore, err := store.Get(ctx, project.OwnerID, "audit-draft")
	if err != nil {
		t.Fatal(err)
	}
	if err := store.InvalidateDeletedRunProjections(ctx, []string{"audit-finalizing", "audit-draft"}); err != nil {
		t.Fatal(err)
	}
	finalizingAfter, err := store.Get(ctx, project.OwnerID, "audit-finalizing")
	if err != nil || finalizingAfter.Revision != finalizingBefore.Revision+1 ||
		finalizingAfter.EventSequence != finalizingBefore.EventSequence ||
		!finalizingAfter.UpdatedAt.Equal(finalizingBefore.UpdatedAt) {
		t.Fatalf("finalizing Audit invalidation = (%+v, %v)", finalizingAfter, err)
	}
	draftAfter, err := store.Get(ctx, project.OwnerID, "audit-draft")
	if err != nil || draftAfter.Revision != draftBefore.Revision+1 ||
		draftAfter.EventSequence != draftBefore.EventSequence ||
		!draftAfter.UpdatedAt.After(draftBefore.UpdatedAt) {
		t.Fatalf("draft Audit invalidation = (%+v, %v)", draftAfter, err)
	}
}
