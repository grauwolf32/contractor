package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// TestPostgresListRoundItemsReadsOneBoundedRound covers V368-023 at the
// realistic bound: an Audit with a full 10,000-item first Round and one more
// item in its second Round holds 10,001 items, but each Round reads only its
// own items, never more than LIM-AUD-26.
func TestPostgresListRoundItemsReadsOneBoundedRound(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-round-items", OwnerID: "owner-round-items", Kind: projectstore.KindProject,
		Name: "Round items", IdempotencyKey: "project-round-items", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := NewPostgresStore(pool)
	audit, _, err := store.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-round-items", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         ProfileIdentity{Name: "profile", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"profile"}`),
		InputSelection:  json.RawMessage(`{"inputs":{}}`),
		Limits: Limits{
			MaxRounds: 2, BatchSize: 1, MaxItemsPerRound: config.MaxAuditItemsPerRound,
			MaxItemsTotal: config.MaxAuditItemsPerRound + 1, MaxSubmittedRuns: config.MaxAuditItemsPerRound + 1,
			MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
		},
		IdempotencyKey: "audit-round-items", RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	first := make([]MaterializedItem, config.MaxAuditItemsPerRound)
	for index := range first {
		key := fmt.Sprintf("check-%05d", index)
		first[index] = MaterializedItem{
			ItemID: "item-" + key, ItemKey: key, Ordinal: index, Kind: "checklist", SubjectKey: key,
			Task: testExact("audit-round-items", "task-"+key, "r1"), Origin: testOrigin(key),
			WorkflowRole: "check", InitialState: ItemReady, Coverage: emptyCoverage(),
		}
	}
	audit, _, err = store.MaterializeRound(ctx, MaterializeRoundParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: "round-one", RoundOrdinal: 1, Manifest: testExact("audit-round-items", "round-one", "r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":{},"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), Items: first,
		IdempotencyKey: "start-round-items", RequestDigest: testDigest("4"),
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, ClaimParams{HolderID: "controller-round-items", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audit_items SET state = 'settled', final_disposition = 'excluded' WHERE round_id = 'round-one'`); err != nil {
		t.Fatal(err)
	}
	closed := closeAuditRoundForTest(t, ctx, store, claims[0], "round-one", 1)
	proposal := testExact("audit-findings", "proposal-round-items", "proposal-r1")
	proposal.MediaType, proposal.SizeBytes = "application/json", 128
	insertAuditProposalHoldForRoundTest(t, ctx, pool, audit, proposal, "proposal-receipt-round-items")
	audit, err = store.Get(ctx, project.OwnerID, audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.AcceptNextRound(ctx, AcceptRoundParams{
		Claim: claims[0], ExpectedAuditRevision: audit.Revision,
		PreviousRoundID: closed.RoundID, RoundID: "round-two", RoundOrdinal: 2,
		Manifest: testExact("audit-round-items", "round-two", "r1"),
		Items: []MaterializedItem{{
			ItemID: "item-round-two", ItemKey: "verify-proposal", Ordinal: 0,
			Kind: "finding-verification", SubjectKey: "subject-round-two",
			Task: testExact("audit-round-items", "round-two-task", "r1"), Origin: testOrigin("verify-proposal"),
			WorkflowRole: "verify", InitialState: ItemAwaitingReview, Coverage: emptyCoverage(),
			ApprovalKind: ItemApprovalActiveCheck, ApprovalDigest: testDigest("7"),
			ProposalSources: []ProposalItemSource{{
				ReceiptID: "proposal-receipt-round-items", ProposedCheckOrdinal: 0, Proposal: proposal,
			}},
		}},
	}); err != nil {
		t.Fatal(err)
	}

	var total int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM audit_items WHERE audit_id = $1`, audit.AuditID).Scan(&total); err != nil ||
		total != config.MaxAuditItemsPerRound+1 {
		t.Fatalf("Audit items = (%d, %v), want %d", total, err, config.MaxAuditItemsPerRound+1)
	}
	second, err := store.ListRoundItems(ctx, audit.AuditID, "round-two")
	if err != nil || len(second) != 1 || second[0].ItemID != "item-round-two" || second[0].RoundID != "round-two" {
		t.Fatalf("second Round items = (%d, %v)", len(second), err)
	}
	firstItems, err := store.ListRoundItems(ctx, audit.AuditID, "round-one")
	if err != nil || len(firstItems) != config.MaxAuditItemsPerRound {
		t.Fatalf("first Round items = (%d, %v), want %d", len(firstItems), err, config.MaxAuditItemsPerRound)
	}
	for index, item := range firstItems {
		if item.Ordinal != index || item.RoundID != "round-one" {
			t.Fatalf("first Round item %d = %+v", index, item)
		}
	}
	if _, err := store.ListRoundItems(ctx, audit.AuditID, "round\x00"); !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid Round ID = %v", err)
	}
}

// TestPostgresReviewDecisionCheckViolationIsInvalid covers V368-023: a review
// decision the stored constraints refuse is ErrInvalid, never an
// unclassified storage error.
func TestPostgresReviewDecisionCheckViolationIsInvalid(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	err := NewPostgresStore(pool).RecordReviewDecision(ctx, ReviewDecisionWriteParams{
		DecisionID: "decision-invalid-key", RequestID: "request-invalid-key", AuditID: "audit-invalid-key",
		ActorID: "owner", Action: "approve", Rationale: "The exact action is approved.",
		SubjectRevision: 1, SubjectDigest: testDigest("a"),
		IdempotencyKey: "key with spaces", RequestDigest: testDigest("b"),
	})
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("review decision with an invalid key = %v, want ErrInvalid", err)
	}
}
