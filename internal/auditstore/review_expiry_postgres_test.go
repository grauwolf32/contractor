package auditstore

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresSettleUndispatchedExpiresItemActionReviews(t *testing.T) {
	for _, test := range []struct {
		name        string
		closing     AuditState
		terminal    AuditState
		disposition FinalDisposition
	}{
		{"finalizing", AuditFinalizing, AuditCompleted, FinalExcluded},
		{"cancelling", AuditCancelling, AuditCancelled, FinalExecutionCancelled},
	} {
		t.Run(test.name, func(t *testing.T) {
			databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
			if databaseURL == "" {
				t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
			}
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool := isolatedAuditPool(t, ctx, databaseURL)
			project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
				ProjectID: "project-review-expiry-" + test.name,
				OwnerID:   "owner-review-expiry-" + test.name,
				Kind:      projectstore.KindProject, Name: "Review expiry",
				IdempotencyKey: "project", RequestDigest: testDigest("1"),
			})
			if err != nil {
				t.Fatal(err)
			}
			store := NewPostgresStore(pool)
			audit, _, err := store.CreateDraft(ctx, CreateDraftParams{
				AuditID: "audit-review-expiry-" + test.name,
				OwnerID: project.OwnerID, ProjectID: project.ProjectID,
				Profile:         ProfileIdentity{Name: "checklist", Version: "1", Digest: testDigest("2")},
				ProfileSnapshot: json.RawMessage(`{"name":"checklist"}`),
				InputSelection:  json.RawMessage(`{}`),
				Limits: Limits{
					MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 2, MaxItemsTotal: 2,
					MaxSubmittedRuns: 2, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
				},
				IdempotencyKey: "audit", RequestDigest: testDigest("3"),
			})
			if err != nil {
				t.Fatal(err)
			}
			items := []MaterializedItem{
				{
					ItemID: "item-one", ItemKey: "check-one", Ordinal: 0, Kind: "checklist",
					SubjectKey: "subject-one", Task: testExact("audits", "task-one", "r1"),
					Origin: testOrigin("check-one"), WorkflowRole: "check",
					InitialState: ItemAwaitingReview, ApprovalKind: ItemApprovalActiveCheck,
					ApprovalDigest: testDigest("4"), Coverage: emptyCoverage(),
				},
				{
					ItemID: "item-two", ItemKey: "check-two", Ordinal: 1, Kind: "checklist",
					SubjectKey: "subject-two", Task: testExact("audits", "task-two", "r1"),
					Origin: testOrigin("check-two"), WorkflowRole: "check",
					InitialState: ItemAwaitingReview, ApprovalKind: ItemApprovalApplicability,
					ApprovalDigest: testDigest("5"), Coverage: emptyCoverage(),
				},
			}
			audit, _, err = store.MaterializeRound(ctx, MaterializeRoundParams{
				OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
				RoundID: "round-review-expiry", RoundOrdinal: 1,
				Manifest:         testExact("audits", "worklist", "r1"),
				BaselineSnapshot: json.RawMessage(`{"inputs":[],"skills":[]}`),
				DeadlineAt:       time.Now().Add(time.Hour), Items: items,
				IdempotencyKey: "start", RequestDigest: testDigest("6"),
			})
			if err != nil {
				t.Fatal(err)
			}
			var pending int
			if err := pool.QueryRow(ctx, `
SELECT count(*) FROM audit_review_requests
 WHERE audit_id = $1 AND subject_kind = 'audit-item-action' AND state = 'pending'`,
				audit.AuditID).Scan(&pending); err != nil || pending != 2 {
				t.Fatalf("pending item reviews = (%d, %v), want 2", pending, err)
			}
			claims, err := store.Claim(ctx, ClaimParams{
				HolderID: "controller-review-expiry", Lease: time.Minute, Limit: 1,
			})
			if err != nil || len(claims) != 1 {
				t.Fatalf("claim Audit = (%+v, %v)", claims, err)
			}
			claim := claims[0]
			if test.closing == AuditFinalizing {
				audit, err = store.TransitionClaimed(ctx, ClaimedTransitionParams{
					Claim: claim, ExpectedRevision: audit.Revision,
					ExpectedState: AuditActive, TargetState: test.closing,
				})
			} else {
				var changed bool
				audit, changed, err = store.Transition(ctx, TransitionParams{
					OwnerID: audit.OwnerID, AuditID: audit.AuditID,
					ExpectedRevision: audit.Revision,
					ExpectedState:    AuditActive, TargetState: test.closing,
					IdempotencyKey: "cancel", RequestDigest: testDigest("7"),
				})
				if err == nil && !changed {
					t.Fatal("cancel transition did not change Audit")
				}
			}
			if err != nil {
				t.Fatal(err)
			}
			settled, err := store.SettleUndispatched(ctx, claim, MaxReconcileRows)
			if err != nil || settled != 2 {
				t.Fatalf("settle undispatched items = (%d, %v)", settled, err)
			}
			settledAudit, err := store.Get(ctx, audit.OwnerID, audit.AuditID)
			if err != nil || settledAudit.Revision != audit.Revision+3 ||
				settledAudit.Revision != settledAudit.EventSequence {
				t.Fatalf("settled Audit revision and events = (%+v, %v)", settledAudit, err)
			}
			if repeated, err := store.SettleUndispatched(ctx, claim, MaxReconcileRows); err != nil || repeated != 0 {
				t.Fatalf("repeat settlement = (%d, %v)", repeated, err)
			}
			storedItems, err := store.ListItems(ctx, audit.AuditID)
			if err != nil || len(storedItems) != 2 {
				t.Fatalf("settled items = (%+v, %v)", storedItems, err)
			}
			for _, item := range storedItems {
				if item.State != ItemSettled || item.FinalDisposition == nil ||
					*item.FinalDisposition != test.disposition {
					t.Errorf("item after settlement = %+v", item)
				}
			}
			terminal, err := store.TransitionClaimed(ctx, ClaimedTransitionParams{
				Claim: claim, ExpectedRevision: settledAudit.Revision,
				ExpectedState: test.closing, TargetState: test.terminal,
			})
			if err != nil || terminal.State != test.terminal || terminal.Revision != terminal.EventSequence {
				t.Fatalf("terminal Audit = (%+v, %v)", terminal, err)
			}
			rows, err := pool.Query(ctx, `
SELECT request_id, state, revision FROM audit_review_requests
 WHERE audit_id = $1 AND subject_kind = 'audit-item-action' ORDER BY request_id`, audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			defer rows.Close()
			seen := make(map[string]bool)
			for rows.Next() {
				var id, state string
				var revision int
				if err := rows.Scan(&id, &state, &revision); err != nil {
					t.Fatal(err)
				}
				if state != "expired" || revision != 2 || seen[id] {
					t.Errorf("item review %s = (%s, revision %d)", id, state, revision)
				}
				seen[id] = true
			}
			if err := rows.Err(); err != nil {
				t.Fatal(err)
			}
			if len(seen) != 2 {
				t.Fatalf("item reviews = %+v, want two", seen)
			}
			events, err := store.ListEvents(ctx, audit.AuditID, 0, MaxPageSize)
			if err != nil {
				t.Fatal(err)
			}
			expired := make(map[string]int)
			for _, event := range events {
				if event.Kind == "review.expired" {
					expired[event.EntityID]++
				}
			}
			if len(expired) != 2 {
				t.Fatalf("review.expired events = %+v", expired)
			}
			for id := range seen {
				if expired[id] != 1 {
					t.Errorf("review.expired event count for %s = %d", id, expired[id])
				}
			}
		})
	}
}
