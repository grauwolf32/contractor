//go:build integration

package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresProposedMembershipHasNoReadOrReviewAuthority(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	owner := "owner-proposed-visibility"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "project-" + owner, OwnerID: owner, Kind: projectstore.KindProject, Name: "Visibility", IdempotencyKey: "project", RequestDigest: testDigest("1")}); err != nil {
		t.Fatal(err)
	}
	f := itemScaleFixture{ctx: ctx, pool: pool, store: newTestStore(pool), owner: owner}
	auditID := f.materialize(t, owner, "audit-proposed-visibility", 4, 4)
	audit, err := f.store.Get(ctx, owner, auditID)
	if err != nil {
		t.Fatal(err)
	}
	proposal := testExact("audit-findings", "visibility-source", "proposal-r1")
	proposal.MediaType, proposal.SizeBytes = "application/json", 128
	receiptID := "receipt-proposed-visibility"
	insertAuditProposalHoldForRoundTest(t, ctx, pool, audit, proposal, receiptID)
	ref, err := json.Marshal(proposal.Ref)
	if err != nil {
		t.Fatal(err)
	}
	stagedRound, stagedItem := "round-private-proposed", "item-private-proposed"
	originalItem := "item-" + auditID + "-00003"
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	cloneWhere(t, ctx, tx, "audit_rounds", "round_id=$1", *audit.CurrentRoundID, 1, map[string]string{"round_id": "'" + stagedRound + "'", "ordinal": "2", "state": "'proposed'", "expected_item_count": "1"})
	cloneWhere(t, ctx, tx, "audit_items", "item_id=$1", originalItem, 1, map[string]string{
		"item_id": "'" + stagedItem + "'", "round_id": "'" + stagedRound + "'", "ordinal": "0", "kind": "'finding-verification'",
		"proposal_receipt_id": "'" + receiptID + "'", "proposal_check_ordinal": "0", "proposal_ref": "'" + string(ref) + "'::jsonb", "proposal_digest": "'" + proposal.Digest + "'",
	})
	cloneWhere(t, ctx, tx, "audit_review_requests", "subject_id=$1", originalItem, 1, map[string]string{"request_id": "src.request_id || '-private'", "subject_id": "'" + stagedItem + "'", "idempotency_key": "src.idempotency_key || '-private'"})
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	query := f.store.QueryStore
	if _, err := query.GetRound(ctx, auditID, stagedRound); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("public proposed Round=%v", err)
	}
	if internal, err := query.getRound(ctx, auditID, stagedRound); err != nil || internal.State != auditstore.RoundProposed {
		t.Fatalf("admission cannot read private Round: %+v,%v", internal, err)
	}
	if rounds, err := query.ListRounds(ctx, auditID); err != nil || len(rounds) != 1 {
		t.Fatalf("visible Rounds=%d,%v", len(rounds), err)
	}
	if count, err := query.CountItems(ctx, auditID); err != nil || count != 4 {
		t.Fatalf("accepted item count=%d,%v", count, err)
	}
	if used, err := query.ScheduledProposalChecks(ctx, auditID, []string{receiptID}); err != nil || len(used) != 0 {
		t.Fatalf("unaccepted source consumed: %v,%v", used, err)
	}
	if items, err := query.ListRoundItems(ctx, auditID, stagedRound); err != nil || len(items) != 0 {
		t.Fatalf("private Round items=%d,%v", len(items), err)
	}
	if items, err := query.ListItemsByIDs(ctx, auditID, []string{stagedItem}); err != nil || len(items) != 0 {
		t.Fatalf("private selected items=%d,%v", len(items), err)
	}
	if items, err := query.ListItemsPage(ctx, auditstore.ListItemsParams{OwnerID: owner, AuditID: auditID, Limit: 20}); err != nil || len(items) != 4 {
		t.Fatalf("public items=%d,%v", len(items), err)
	}
	if coverage, err := query.ListCoverage(ctx, auditID, stagedRound, -1, 20); err != nil || len(coverage) != 0 {
		t.Fatalf("private coverage=%d,%v", len(coverage), err)
	}
	if report, err := query.ListReportItems(ctx, auditID, stagedRound, -1, 20); err != nil || len(report) != 0 {
		t.Fatalf("private report items=%d,%v", len(report), err)
	}
	claims, err := f.store.Claim(ctx, auditstore.ClaimParams{HolderID: "visibility-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim=%v,%v", claims, err)
	}
	claim := claims[0]
	if snapshot, err := f.store.GetReconcileSnapshot(ctx, claim); err != nil || len(snapshot.Items) != 4 {
		t.Fatalf("reconcile items=%d,%v", len(snapshot.Items), err)
	}
	if ready, err := f.store.ClaimStore.listReadyItems(ctx, auditID, stagedRound); err != nil || len(ready) != 0 {
		t.Fatalf("private dispatch candidates=%d,%v", len(ready), err)
	}
	if attempts, err := f.store.NextItemAttempts(ctx, claim, []string{stagedItem}); err != nil || len(attempts) != 0 {
		t.Fatalf("private attempts=%v,%v", attempts, err)
	}
	if awaiting, err := f.store.ClaimStore.onlyAwaitingReview(ctx, auditID); err != nil || awaiting {
		t.Fatalf("private item influenced waiting=%v,%v", awaiting, err)
	}
	params := auditstore.ReviewListParams{OwnerID: owner, AuditID: auditID, Limit: 20}
	if reviews, err := query.ListReviews(ctx, params); err != nil || len(reviews) != 1 {
		t.Fatalf("visible reviews=%d,%v", len(reviews), err)
	}
	if basis, err := query.ReviewPageBasis(ctx, params); err != nil || basis.Total != 1 {
		t.Fatalf("review basis=%+v,%v", basis, err)
	}
	if reviews, err := query.ListOwnerReviews(ctx, auditstore.OwnerReviewListParams{OwnerID: owner, Limit: 20}); err != nil || len(reviews) != 1 {
		t.Fatalf("owner reviews=%d,%v", len(reviews), err)
	}
	if workspace, err := query.GetWorkspace(ctx, owner, auditID); err != nil || workspace.PendingReviews != 1 || workspace.TotalChecks != 4 {
		t.Fatalf("workspace=%+v,%v", workspace, err)
	}
	var requestID string
	if err := pool.QueryRow(ctx, `SELECT request_id FROM audit_review_requests WHERE audit_id=$1 AND subject_id=$2`, auditID, stagedItem).Scan(&requestID); err != nil {
		t.Fatal(err)
	}
	if _, err := query.GetReview(ctx, owner, auditID, requestID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("private direct review=%v", err)
	}
	tx, err = pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewReviewTxStore(tx, artifactpgtest.NewStorage()).LockActionReview(ctx, owner, auditID, requestID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("private review lock=%v", err)
	}
	if err := NewReviewStore(tx, artifactpgtest.NewStorage()).ApplyItemReviewDecision(ctx, auditID, stagedItem, string(auditstore.ItemApprovalActiveCheck), testDigest("5"), "approve", ""); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("private item decision=%v", err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	var live int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM audit_live_item_reviews WHERE audit_id=$1 AND item_id=$2`, auditID, stagedItem).Scan(&live); err != nil || live != 0 {
		t.Fatalf("private live authority=%d,%v", live, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audit_review_requests SET expires_at='2000-01-01' WHERE request_id=$1`, requestID); err != nil {
		t.Fatal(err)
	}
	if changed, err := f.store.RenewExpiredItemReview(ctx, claim, audit.Revision); err != nil || changed {
		t.Fatalf("private review renewed=%v,%v", changed, err)
	}
	// Ready construction rows must also be invisible to actual dispatch and
	// attempt selection, independently of their action-review state.
	if _, err := pool.Exec(ctx, `UPDATE audit_items SET state='ready' WHERE item_id=$1`, stagedItem); err != nil {
		t.Fatal(err)
	}
	if ready, err := f.store.ClaimStore.listReadyItems(ctx, auditID, stagedRound); err != nil || len(ready) != 0 {
		t.Fatalf("ready private dispatch candidates=%d,%v", len(ready), err)
	}
	if attempts, err := f.store.NextItemAttempts(ctx, claim, []string{stagedItem}); err != nil || len(attempts) != 0 {
		t.Fatalf("ready private attempts=%v,%v", attempts, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audits SET state='waiting_review' WHERE audit_id=$1`, auditID); err != nil {
		t.Fatal(err)
	}
	if err := f.store.ReleaseClaim(ctx, claim); err != nil {
		t.Fatal(err)
	}
	if claims, err := f.store.Claim(ctx, auditstore.ClaimParams{HolderID: "must-not-wake", Lease: time.Minute, Limit: 1}); err != nil || len(claims) != 0 {
		t.Fatalf("private expired review woke controller: %v,%v", claims, err)
	}
}

func TestPostgresAcceptedItemCountReadsRoundCountersAtAuditCeiling(t *testing.T) {
	f := newItemScaleFixture(t)
	if count, err := f.store.CountItems(f.ctx, f.auditID); err != nil || count != itemScaleItems {
		t.Fatalf("accepted count=%d,%v", count, err)
	}
	for _, mode := range explaintest.Modes {
		plan := explainPrepared(t, f.ctx, f.pool, mode, countAcceptedItemsSQL, f.auditID)
		if items, rounds := plan.Reads.Of("audit_items"), plan.Reads.Of("audit_rounds"); items != 0 || rounds > itemScaleRounds+planningEntries {
			t.Fatalf("%s read items=%d rounds=%d", mode, items, rounds)
		}
	}
}
