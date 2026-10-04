package auditservice

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestAuditReportReviewExpiresOnCancelOrDelete(t *testing.T) {
	for _, action := range []string{"cancel", "delete"} {
		t.Run(action, func(t *testing.T) {
			databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
			if databaseURL == "" {
				t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
			}
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool := isolatedAuditServicePool(t, ctx, databaseURL)
			project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
				ProjectID: "project-report-close-" + action,
				OwnerID:   "owner-report-close-" + action,
				Kind:      projectstore.KindProject, Name: "Report close",
				IdempotencyKey: "project", RequestDigest: serviceTestDigest("project"),
			})
			if err != nil {
				t.Fatal(err)
			}
			store := auditstore.NewPostgresStore(pool)
			audit, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
				AuditID: "audit-report-close-" + action,
				OwnerID: project.OwnerID, ProjectID: project.ProjectID,
				Profile: auditstore.ProfileIdentity{
					Name: "report-review", Version: "1", Digest: serviceTestDigest("profile"),
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
			findingID := seedAuditFinding(t, ctx, pool, project.ProjectID, project.OwnerID, audit.AuditID, "report-close")
			audit, err = store.Get(ctx, audit.OwnerID, audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			waiting, candidate, claim := proposeReportForTest(t, ctx, pool, store, audit, "close-"+action)
			requestID := candidate.RequestID
			intake, err := findingintake.New(pool)
			if err != nil {
				t.Fatal(err)
			}
			service := &Service{pool: pool, findings: intake, now: time.Now}
			before, err := service.GetWorkspace(ctx, audit.OwnerID, audit.AuditID)
			if err != nil || before.PendingReviews != 1 {
				t.Fatalf("workspace before closing report review = (%+v, %v)", before, err)
			}
			mutation := MutationParams{
				OwnerID: audit.OwnerID, AuditID: audit.AuditID,
				ExpectedRevision: waiting.Revision, IdempotencyKey: action,
				RequestDigest: serviceTestDigest(action),
			}
			var result MutationResult
			if action == "cancel" {
				result, err = service.Cancel(ctx, mutation)
			} else {
				result, err = service.Delete(ctx, mutation)
			}
			if err != nil || result.Audit.State != auditstore.AuditCancelling ||
				result.Audit.Revision != waiting.Revision+2 ||
				result.Audit.Revision != result.Audit.EventSequence {
				t.Fatalf("%s report review = (%+v, %v)", action, result, err)
			}
			review, err := service.GetReview(ctx, audit.OwnerID, audit.AuditID, requestID)
			if err != nil || review.State != ReviewExpired || review.Revision != 2 {
				t.Fatalf("expired report review = (%+v, %v)", review, err)
			}
			events, err := store.ListEvents(ctx, audit.AuditID, 0, auditstore.MaxPageSize)
			if err != nil {
				t.Fatal(err)
			}
			expired := 0
			for _, event := range events {
				if event.Kind == "review.expired" && event.EntityID == requestID {
					expired++
				}
			}
			if expired != 1 {
				t.Fatalf("report review expiry events = %d, want 1", expired)
			}
			if action == "cancel" {
				terminal, err := store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
					Claim: claim, ExpectedRevision: result.Audit.Revision,
					ExpectedState: auditstore.AuditCancelling, TargetState: auditstore.AuditCancelled,
				})
				if err != nil || terminal.Revision != terminal.EventSequence {
					t.Fatalf("cancelled Audit = (%+v, %v)", terminal, err)
				}
				workspace, err := service.GetWorkspace(ctx, audit.OwnerID, audit.AuditID)
				if err != nil || workspace.PendingReviews != 0 {
					t.Fatalf("terminal pending decisions = (%+v, %v)", workspace, err)
				}
				created, err := service.CreateFindingReview(ctx, CreateFindingReviewParams{
					OwnerID: audit.OwnerID, AuditID: audit.AuditID, FindingID: findingID,
					ExpectedRevision: 1, RequestID: "review-finding-after-cancel",
					IdempotencyKey: "review-finding-after-cancel",
					RequestDigest:  serviceTestDigest("finding-review"),
				})
				if err != nil {
					t.Fatalf("create finding review after report cancellation: %v", err)
				}
				if _, err := service.DecideFinding(ctx, DecideFindingParams{
					OwnerID: audit.OwnerID, AuditID: audit.AuditID,
					RequestID:               created.Request.RequestID,
					ExpectedRequestRevision: created.Request.Revision,
					DecisionID:              "decision-finding-after-cancel", Verdict: VerdictNeedsEvidence,
					Rationale:      "More evidence is needed.",
					IdempotencyKey: "decision-finding-after-cancel",
					RequestDigest:  serviceTestDigest("finding-decision"),
				}); err != nil {
					t.Fatalf("decide finding review after report cancellation: %v", err)
				}
			}
		})
	}
}

// proposeReportForTest finalizes a draft Audit through an empty Round into a
// human report review of a frozen candidate. The returned claim stays live.
func proposeReportForTest(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, store *auditstore.PostgresStore,
	audit auditstore.Audit, suffix string,
) (auditstore.Audit, auditstore.ReportCandidate, auditstore.ControllerClaim) {
	t.Helper()
	revision := "r1"
	roundID := "round-report-" + suffix
	audit, _, err := store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: roundID, RoundOrdinal: 1,
		Manifest: auditstore.ExactArtifact{
			Ref: contracts.ArtifactRef{
				Namespace: "audit-reports", Name: "worklist", Revision: &revision,
			},
			Digest: serviceTestDigest("worklist"),
		},
		BaselineSnapshot: json.RawMessage(`{"inputs":[],"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), Items: []auditstore.MaterializedItem{},
		IdempotencyKey: "start", RequestDigest: serviceTestDigest("start"),
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{
		HolderID: "report-" + suffix + "-controller", Lease: time.Minute, Limit: 1,
	})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	finalizing, err := store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
		Claim: claim, ExpectedRevision: audit.Revision,
		ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing,
	})
	if err != nil {
		t.Fatal(err)
	}
	waiting, err := store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
		Claim: claim, ExpectedRevision: finalizing.Revision,
		ExpectedState: auditstore.AuditFinalizing, TargetState: auditstore.AuditWaitingReview,
	})
	if err != nil {
		t.Fatal(err)
	}
	requestID := "review-report-" + suffix
	digest := serviceTestDigest("report-candidate")
	link := func(key, name, media string) json.RawMessage {
		encoded, err := json.Marshal(auditstore.ArtifactLink{
			LogicalKey: key,
			Artifact: auditstore.ExactArtifact{
				Ref: contracts.ArtifactRef{
					Namespace: "audit-reports", Name: name, Revision: &revision,
				},
				Digest: serviceTestDigest(name), MediaType: media, SizeBytes: 16,
			},
			SourceProvenance: json.RawMessage(`{"schema":"contractor.audit.report-provenance.v1"}`),
		})
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO audit_review_requests (
    request_id, audit_id, finding_id, subject_kind, subject_id, kind,
    subject_revision, subject_digest, requested_actions, state,
    expires_at, idempotency_key, request_digest
) VALUES (
    $1, $2, NULL, 'audit-report', $2, 'report-acceptance',
    $3, $4, '["approve","reject"]'::jsonb, 'pending',
    clock_timestamp() + interval '30 days', 'report-review', $4
)`, requestID, audit.AuditID, finalizing.Revision, digest); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO audit_report_candidates (
    audit_id, request_id, round_id, subject_revision, subject_digest,
    machine_link, summary_link
) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb)`,
		audit.AuditID, requestID, roundID, finalizing.Revision, digest,
		link(auditstore.ReportMachineLogicalKey, "report.json", "application/json"),
		link(auditstore.ReportSummaryLogicalKey, "report.md", "text/markdown")); err != nil {
		t.Fatal(err)
	}
	candidate, err := store.GetReportCandidate(ctx, audit.AuditID)
	if err != nil || candidate.RequestID != requestID {
		t.Fatalf("report candidate = (%+v, %v)", candidate, err)
	}
	return waiting, candidate, claim
}
