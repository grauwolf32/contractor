package auditservice

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestAuditPurgeRemovesDecidedReviews deletes and purges Audits after their
// owner decided an item action, a finding with collected assessments, or a
// proposed report. Immutable decisions and assessments leave with the Audit.
func TestAuditPurgeRemovesDecidedReviews(t *testing.T) {
	for _, subject := range []string{"action", "finding", "report"} {
		t.Run(subject, func(t *testing.T) {
			databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
			if databaseURL == "" {
				t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
			}
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool := isolatedAuditServicePool(t, ctx, databaseURL)
			project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
				ProjectID: "project-purge-" + subject, OwnerID: "owner-purge-" + subject,
				Kind: projectstore.KindProject, Name: "Decided purge",
				IdempotencyKey: "project", RequestDigest: serviceTestDigest("project"),
			})
			if err != nil {
				t.Fatal(err)
			}
			store := auditstore.NewPostgresStore(pool)
			audit, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
				AuditID: "audit-purge-" + subject, OwnerID: project.OwnerID, ProjectID: project.ProjectID,
				Profile: auditstore.ProfileIdentity{
					Name: "decided-purge", Version: "1", Digest: serviceTestDigest("profile"),
				},
				ProfileSnapshot: json.RawMessage(`{"interaction":{"reportAcceptance":"human-required"}}`),
				InputSelection:  json.RawMessage(`{}`),
				Limits: auditstore.Limits{
					MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
					MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
				},
				IdempotencyKey: "audit", RequestDigest: serviceTestDigest("audit"),
			})
			if err != nil {
				t.Fatal(err)
			}
			intake, err := findingintake.New(pool)
			if err != nil {
				t.Fatal(err)
			}
			service := &Service{pool: pool, findings: intake, now: time.Now}
			switch subject {
			case "action":
				rejectItemActionForPurge(t, ctx, store, service, audit)
			case "finding":
				findingID := seedAuditFinding(t, ctx, pool, project.ProjectID, project.OwnerID, audit.AuditID, "purge")
				decideFindingForTest(t, ctx, service, project.OwnerID, audit.AuditID, findingID, 1,
					"purge", VerdictTruePositive, reviewStringPointer(string(SeverityHigh)), nil)
				seedAuditFindingAttemptHistory(t, ctx, pool, audit.AuditID, findingID, "receipt-purge")
			case "report":
				rejectReportForPurge(t, ctx, pool, store, service, audit)
			}
			current, err := store.Get(ctx, audit.OwnerID, audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			deleted, err := service.Delete(ctx, MutationParams{
				OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: current.Revision,
				IdempotencyKey: "delete", RequestDigest: serviceTestDigest("delete"),
			})
			if err != nil {
				t.Fatal(err)
			}
			claim := claimAuditForPurge(t, ctx, store, audit.AuditID)
			if deleted.Audit.State == auditstore.AuditCancelling {
				if _, err := store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
					Claim: claim, ExpectedRevision: deleted.Audit.Revision,
					ExpectedState: auditstore.AuditCancelling, TargetState: auditstore.AuditDeleting,
					Reason: deleted.Audit.StopReason,
				}); err != nil {
					t.Fatal(err)
				}
				if _, released, err := store.ReleaseDispatchHold(ctx, claim); err != nil || !released {
					t.Fatalf("release dispatch hold = (%t, %v)", released, err)
				}
			}
			var decisions int
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM audit_review_decisions`).Scan(&decisions); err != nil || decisions == 0 {
				t.Fatalf("decisions before purge = (%d, %v)", decisions, err)
			}
			if err := store.PurgeClaimed(ctx, claim, auditdomain.ArtifactNamespace(audit.AuditID)); err != nil {
				t.Fatalf("purge Audit with a decided %s review: %v", subject, err)
			}
			for _, table := range []string{"audits", "audit_review_requests", "audit_review_decisions", "audit_finding_assessments"} {
				var remaining int
				if err := pool.QueryRow(ctx, `SELECT count(*) FROM `+table).Scan(&remaining); err != nil || remaining != 0 {
					t.Fatalf("%s after purge = (%d, %v)", table, remaining, err)
				}
			}
		})
	}
}

// rejectItemActionForPurge starts the Audit with one item awaiting active-check
// approval and rejects that exact action through the review API.
func rejectItemActionForPurge(
	t *testing.T, ctx context.Context, store *auditstore.PostgresStore, service *Service, audit auditstore.Audit,
) {
	t.Helper()
	revision := "r1"
	source := contracts.ArtifactRef{Namespace: "inputs", Name: "checks", Revision: &revision}
	item := auditstore.MaterializedItem{
		ItemID: "item-purge", ItemKey: "purge-check", Ordinal: 0, Kind: "checklist",
		SubjectKey: "purge-subject",
		Task: auditstore.ExactArtifact{
			Ref:    contracts.ArtifactRef{Namespace: "audit-tasks", Name: "purge-check", Revision: &revision},
			Digest: serviceTestDigest("task"), MediaType: "application/zip", SizeBytes: 1,
		},
		Origin: auditstore.ItemOrigin{
			Schema: auditstore.ItemOriginSchema, SourceRef: &source,
			SourceContentDigest: serviceTestDigest("source"), SourceMediaType: "application/json",
			CanonicalInventoryDigest: serviceTestDigest("inventory"), EntryKey: "purge-check", EntryVersion: "1",
		},
		WorkflowRole: "check", InitialState: auditstore.ItemAwaitingReview,
		ApprovalKind: auditstore.ItemApprovalActiveCheck, ApprovalDigest: serviceTestDigest("approval"),
		Coverage: auditstore.Coverage{
			Status: auditstore.CoverageNotTested, Requested: []string{}, Completed: []string{}, Gaps: []string{},
		},
	}
	if _, _, err := store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: "round-purge", RoundOrdinal: 1,
		Manifest: auditstore.ExactArtifact{
			Ref:    contracts.ArtifactRef{Namespace: "audit-rounds", Name: "worklist", Revision: &revision},
			Digest: serviceTestDigest("worklist"),
		},
		BaselineSnapshot: json.RawMessage(`{"inputs":[],"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), Items: []auditstore.MaterializedItem{item},
		IdempotencyKey: "start", RequestDigest: serviceTestDigest("start"),
	}); err != nil {
		t.Fatal(err)
	}
	decision, err := service.DecideActionReview(ctx, DecideActionReviewParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, RequestID: "review-" + item.ItemID,
		ExpectedRequestRevision: 1, DecisionID: "decision-purge-reject", Action: ReviewReject,
		Rationale:      "The exact active check is outside this assessment.",
		IdempotencyKey: "decision-purge-reject", RequestDigest: serviceTestDigest("reject"),
	})
	if err != nil || decision.Request.State != ReviewDecided {
		t.Fatalf("reject item action = (%+v, %v)", decision, err)
	}
}

// rejectReportForPurge finalizes the Audit into a human report review and
// rejects the frozen candidate, which fails the Audit.
func rejectReportForPurge(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	store *auditstore.PostgresStore, service *Service, audit auditstore.Audit,
) {
	t.Helper()
	waiting, candidate, claim := proposeReportForTest(t, ctx, pool, store, audit, "purge")
	decision, err := service.DecideActionReview(ctx, DecideActionReviewParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, RequestID: candidate.RequestID,
		ExpectedRequestRevision: 1, DecisionID: "decision-purge-report", Action: ReviewReject,
		Rationale:      "The proposed report is incomplete.",
		IdempotencyKey: "decision-purge-report", RequestDigest: serviceTestDigest("report"),
	})
	if err != nil || decision.Request.State != ReviewDecided {
		t.Fatalf("reject report candidate = (%+v, %v)", decision, err)
	}
	failed, err := store.Get(ctx, audit.OwnerID, audit.AuditID)
	if err != nil || failed.State != auditstore.AuditFailed || failed.Revision <= waiting.Revision {
		t.Fatalf("Audit after report rejection = (%+v, %v)", failed, err)
	}
	if err := store.ReleaseClaim(ctx, claim); err != nil {
		t.Fatal(err)
	}
}

func claimAuditForPurge(
	t *testing.T, ctx context.Context, store *auditstore.PostgresStore, auditID string,
) auditstore.ControllerClaim {
	t.Helper()
	claims, err := store.Claim(ctx, auditstore.ClaimParams{
		HolderID: "purge-controller", Lease: time.Minute, Limit: 1,
	})
	if err != nil || len(claims) != 1 || claims[0].AuditID != auditID {
		t.Fatalf("claim Audit %s = (%+v, %v)", auditID, claims, err)
	}
	return claims[0]
}
