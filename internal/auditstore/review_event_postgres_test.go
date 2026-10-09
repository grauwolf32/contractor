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

func TestPostgresReviewEventKeepsAuditRevisionAndSequenceAtomic(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-review-event", OwnerID: "owner-review-event", Kind: projectstore.KindProject,
		Name: "Review event", IdempotencyKey: "create-project", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := NewPostgresStore(pool)
	audit, _, err := store.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-review-event", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         ProfileIdentity{Name: "review-event", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{}`), InputSelection: json.RawMessage(`{}`),
		Limits: Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024},
		IdempotencyKey: "create-audit", RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	params := ReviewEventParams{
		AuditID: audit.AuditID, Kind: "review.requested", EntityID: "review-one",
		Summary: map[string]any{"subjectKind": "finding"},
	}
	if err := store.AppendReviewEvent(ctx, params); err != nil {
		t.Fatal(err)
	}
	invalid := params
	invalid.Kind = "INVALID"
	if err := store.AppendReviewEvent(ctx, invalid); err == nil {
		t.Fatal("invalid event was recorded")
	}
	current, err := store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil || current.Revision != audit.Revision+1 || current.EventSequence != audit.EventSequence+1 {
		t.Fatalf("Audit after rejected event = (%+v, %v)", current, err)
	}
	params.Kind, params.EntityID = "review.decided", "decision-one"
	if err := store.AppendReviewEvent(ctx, params); err != nil {
		t.Fatal(err)
	}
	current, err = store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil || current.Revision != audit.Revision+2 || current.EventSequence != audit.EventSequence+2 {
		t.Fatalf("Audit after accepted events = (%+v, %v)", current, err)
	}
	events, err := store.ListEvents(ctx, audit.AuditID, audit.EventSequence, 2)
	if err != nil || len(events) != 2 || events[0].Kind != "review.requested" ||
		events[1].Kind != "review.decided" ||
		events[1].Sequence != events[0].Sequence+1 {
		t.Fatalf("contiguous review events = (%+v, %v)", events, err)
	}
	beforeBatch := current
	batch := []ReviewEventParams{params, invalid}
	if err := store.AppendReviewEvents(ctx, batch); err == nil {
		t.Fatal("batch with an invalid second event was recorded")
	}
	current, err = store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil || current.Revision != beforeBatch.Revision || current.EventSequence != beforeBatch.EventSequence {
		t.Fatalf("failed batch partially allocated events: (%+v, %v)", current, err)
	}
	revision := uint64(7)
	batch[1] = ReviewEventParams{AuditID: audit.AuditID, Kind: "review.expired",
		EntityID: "review-batch-two", EntityRevision: &revision, Summary: map[string]any{"subjectKind": "finding"}}
	if err := store.AppendReviewEvents(ctx, batch); err != nil {
		t.Fatal(err)
	}
	current, err = store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil || current.Revision != beforeBatch.Revision+2 || current.EventSequence != beforeBatch.EventSequence+2 {
		t.Fatalf("batch did not advance once per event: (%+v, %v)", current, err)
	}
	events, err = store.ListEvents(ctx, audit.AuditID, beforeBatch.EventSequence, 2)
	if err != nil || len(events) != 2 || events[0].Kind != batch[0].Kind || events[1].Kind != batch[1].Kind ||
		events[1].Sequence != events[0].Sequence+1 || events[1].EntityRevision == nil || *events[1].EntityRevision != revision {
		t.Fatalf("batch order or entity revision changed: (%+v, %v)", events, err)
	}
	params.AuditID = "missing-audit"
	if err := store.AppendReviewEvent(ctx, params); !errors.Is(err, pgx.ErrNoRows) {
		t.Fatalf("missing Audit event error = %v", err)
	}
}
