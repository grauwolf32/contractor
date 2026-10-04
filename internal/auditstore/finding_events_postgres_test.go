package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

func TestPostgresFindingAuditEventsPreserveQuotaAndSequence(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-finding-events", OwnerID: "owner-finding-events", Kind: projectstore.KindProject,
		Name: "Finding events", IdempotencyKey: "create-project", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := NewPostgresStore(pool)
	audit, _, err := store.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-finding-events", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         ProfileIdentity{Name: "finding-events", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{}`), InputSelection: json.RawMessage(`{}`),
		Limits: Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024},
		IdempotencyKey: "create-audit", RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	assessment := DirectFindingAssessmentParams{
		AuditID: audit.AuditID, FindingID: "finding-one", FindingRevision: 2,
		AssessmentID: "assessment-one", RetainedBytes: 1025,
	}
	if err := store.RecordDirectFindingAssessment(ctx, assessment); !errors.Is(err, pgx.ErrNoRows) {
		t.Fatalf("over-quota assessment error = %v", err)
	}
	current, err := store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil || current.Revision != audit.Revision || current.EventSequence != audit.EventSequence ||
		current.RetainedEvidenceBytes != 0 {
		t.Fatalf("Audit after rejected assessment = (%+v, %v)", current, err)
	}
	assessment.RetainedBytes = 2
	if err := store.RecordDirectFindingAssessment(ctx, assessment); err != nil {
		t.Fatal(err)
	}
	badProposal := RejectedFindingProposalParams{
		AuditID: audit.AuditID, ReceiptID: "", RunID: "run-one", Reason: "invalid-proposal",
	}
	if err := store.RecordRejectedFindingProposal(ctx, badProposal); err == nil {
		t.Fatal("invalid rejection event was recorded")
	}
	badProposal.ReceiptID = "receipt-one"
	if err := store.RecordRejectedFindingProposal(ctx, badProposal); err != nil {
		t.Fatal(err)
	}
	current, err = store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil || current.Revision != audit.Revision+2 ||
		current.EventSequence != audit.EventSequence+2 || current.RetainedEvidenceBytes != 2 {
		t.Fatalf("Audit after finding events = (%+v, %v)", current, err)
	}
	events, err := store.ListEvents(ctx, audit.AuditID, audit.EventSequence, 2)
	if err != nil || len(events) != 2 || events[0].Kind != "finding.assessed" ||
		events[1].Kind != "finding.proposal_rejected" || events[1].Sequence != events[0].Sequence+1 {
		t.Fatalf("contiguous finding events = (%+v, %v)", events, err)
	}
}
