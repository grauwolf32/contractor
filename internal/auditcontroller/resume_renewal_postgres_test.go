//go:build integration

package auditcontroller

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

// TestPostgresControllerDispatchesOnlyAfterApprovingReviewRenewedByResume
// lets a pending or approved item review expire while the Audit is paused.
// Resume renews it; neither the stale request nor the Resume itself
// authorizes execution, while an approval of the fresh request does.
func TestPostgresControllerDispatchesOnlyAfterApprovingReviewRenewedByResume(t *testing.T) {
	for _, approved := range []bool{false, true} {
		name := "pending"
		if approved {
			name = "approved"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			harness := newPostgresControllerReviewHarness(t, ctx, 1, 0)
			auditID, ownerID := harness.started.Audit.AuditID, harness.started.Audit.OwnerID
			controller := harness.controller(t)
			reviews, err := harness.service.ListReviews(ctx, auditservice.ReviewListParams{
				OwnerID: ownerID, AuditID: auditID, Limit: 10,
			})
			if err != nil || len(reviews) != 1 {
				t.Fatalf("initial reviews = (%+v, %v)", reviews, err)
			}
			original := reviews[0]
			if approved {
				if _, err := harness.service.DecideActionReview(ctx, auditservice.DecideActionReviewParams{
					OwnerID: ownerID, AuditID: auditID, RequestID: original.RequestID,
					ExpectedRequestRevision: original.Revision, DecisionID: "original-decision",
					Action: auditservice.ReviewApprove, Rationale: "Approve the original exact task.",
					IdempotencyKey: "original-decision", RequestDigest: postgresDigest("original-decision"),
				}); err != nil {
					t.Fatal(err)
				}
			} else {
				for step := 0; step < 4; step++ {
					if _, err := controller.RunOnce(ctx); err != nil {
						t.Fatalf("reach waiting review step %d: %v", step, err)
					}
				}
			}
			audit, err := harness.audits.Get(ctx, ownerID, auditID)
			if err != nil {
				t.Fatal(err)
			}
			paused, err := harness.service.Pause(ctx, auditservice.MutationParams{
				OwnerID: ownerID, AuditID: auditID, ExpectedRevision: audit.Revision,
				IdempotencyKey: "pause-" + name, RequestDigest: postgresDigest("pause-" + name),
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := harness.pool.Exec(ctx, `UPDATE audit_review_requests
SET expires_at=clock_timestamp()-interval '1 second' WHERE request_id=$1`, original.RequestID); err != nil {
				t.Fatal(err)
			}
			unlimited := 0
			resumed, err := harness.service.Resume(ctx, auditservice.MutationParams{
				OwnerID: ownerID, AuditID: auditID, ExpectedRevision: paused.Audit.Revision,
				IdempotencyKey: "resume-" + name, RequestDigest: postgresDigest("resume-" + name),
				DeadlineSeconds: &unlimited,
			})
			if err != nil {
				t.Fatal(err)
			}
			if resumed.Audit.Revision != paused.Audit.Revision+3 || resumed.Audit.Revision != resumed.Audit.EventSequence {
				t.Fatalf("Resume did not record one revision per renewal event: revision %d, sequence %d, paused at %d",
					resumed.Audit.Revision, resumed.Audit.EventSequence, paused.Audit.Revision)
			}
			reviews, err = harness.service.ListReviews(ctx, auditservice.ReviewListParams{
				OwnerID: ownerID, AuditID: auditID, Limit: 10,
			})
			if err != nil || len(reviews) != 2 {
				t.Fatalf("renewed reviews = (%+v, %v)", reviews, err)
			}
			var renewed auditservice.ReviewRequest
			for _, review := range reviews {
				if review.RequestID == original.RequestID {
					if review.State != auditservice.ReviewExpired {
						t.Fatalf("original review after Resume = %+v", review)
					}
				} else {
					renewed = review
				}
			}
			if renewed.State != auditservice.ReviewPending || renewed.ExpiresAt != nil {
				t.Fatalf("renewed review = %+v", renewed)
			}
			for step := 0; step < 4; step++ {
				if _, err := controller.RunOnce(ctx); err != nil {
					t.Fatalf("reconcile renewed review step %d: %v", step, err)
				}
			}
			executions, err := harness.audits.ListExecutions(ctx, auditID)
			if err != nil || len(executions) != 0 {
				t.Fatalf("executions before the renewed approval = (%+v, %v)", executions, err)
			}
			if audit, err = harness.audits.Get(ctx, ownerID, auditID); err != nil ||
				audit.State != auditstore.AuditWaitingReview {
				t.Fatalf("Audit awaiting the renewed review = (%s, %v)", audit.State, err)
			}
			if _, err := harness.service.DecideActionReview(ctx, auditservice.DecideActionReviewParams{
				OwnerID: ownerID, AuditID: auditID, RequestID: renewed.RequestID,
				ExpectedRequestRevision: renewed.Revision, DecisionID: "renewed-decision",
				Action: auditservice.ReviewApprove, Rationale: "Approve the fresh exact task.",
				IdempotencyKey: "renewed-decision", RequestDigest: postgresDigest("renewed-decision"),
			}); err != nil {
				t.Fatalf("decide renewed review: %v", err)
			}
			for step := 0; step < 4 && len(executions) == 0; step++ {
				if _, err := controller.RunOnce(ctx); err != nil {
					t.Fatalf("dispatch after the renewed approval step %d: %v", step, err)
				}
				if executions, err = harness.audits.ListExecutions(ctx, auditID); err != nil {
					t.Fatal(err)
				}
			}
			if len(executions) != 1 {
				t.Fatalf("executions after the renewed approval = %+v", executions)
			}
		})
	}
}
