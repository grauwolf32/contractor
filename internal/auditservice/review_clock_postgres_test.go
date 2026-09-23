package auditservice

import (
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// Review expiry follows the PostgreSQL clock that the Controller and
// execution authorization use, never the owner-facing process clock.
func TestAuditReviewExpiryFollowsDatabaseClock(t *testing.T) {
	ctx, pool, service, started, now := newTimeControlAudit(t, 7200)
	base := *now
	audit := started.Audit
	reviews, err := service.ListReviews(ctx, ReviewListParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, Limit: 10,
	})
	if err != nil || len(reviews) != 1 || reviews[0].SubjectKind != ReviewSubjectItemAction ||
		reviews[0].ExpiresAt == nil || !reviews[0].ExpiresAt.After(time.Now()) {
		t.Fatalf("item review = (%+v, %v)", reviews, err)
	}
	itemReview := reviews[0]

	paused, err := service.Pause(ctx, MutationParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		IdempotencyKey: "pause-clock", RequestDigest: serviceTestDigest("pause-clock"),
	})
	if err != nil {
		t.Fatal(err)
	}
	// A process clock past the request's expiry neither renews it on Resume
	// nor rejects a decision while the database clock is still before it.
	*now = base.Add(3 * time.Hour)
	if _, err := service.Resume(ctx, MutationParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: paused.Audit.Revision,
		IdempotencyKey: "resume-clock", RequestDigest: serviceTestDigest("resume-clock"),
	}); err != nil {
		t.Fatal(err)
	}
	var requests, pending int
	if err := pool.QueryRow(ctx, `
SELECT count(*), count(*) FILTER (WHERE state = 'pending')
  FROM audit_review_requests WHERE audit_id = $1`, audit.AuditID).Scan(&requests, &pending); err != nil {
		t.Fatal(err)
	}
	if requests != 1 || pending != 1 {
		t.Fatalf("Resume renewed an unexpired review: requests=%d pending=%d", requests, pending)
	}
	if _, err := service.DecideActionReview(ctx, DecideActionReviewParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, RequestID: itemReview.RequestID,
		ExpectedRequestRevision: itemReview.Revision, DecisionID: "decision-clock-item",
		Action: ReviewApprove, Rationale: "Approved before the database expiry.",
		IdempotencyKey: "decision-clock-item", RequestDigest: serviceTestDigest("decision-clock-item"),
	}); err != nil {
		t.Fatalf("decide unexpired item review with a skewed clock: %v", err)
	}

	// A process clock far behind the database neither shortens a new
	// request's window nor keeps an expired request decidable.
	*now = base.Add(-30 * 24 * time.Hour)
	findingID := seedAuditFinding(t, ctx, pool, audit.ProjectID, audit.OwnerID, audit.AuditID, "clock")
	review, err := service.CreateFindingReview(ctx, CreateFindingReviewParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, FindingID: findingID, ExpectedRevision: 1,
		RequestID: "review-clock", IdempotencyKey: "review-clock",
		RequestDigest: serviceTestDigest("review-clock"),
	})
	if err != nil || review.Request.ExpiresAt == nil ||
		!review.Request.ExpiresAt.After(time.Now().Add(defaultReviewTTL-time.Hour)) {
		t.Fatalf("finding review window = (%+v, %v)", review.Request, err)
	}
	if _, err := pool.Exec(ctx, `
UPDATE audit_review_requests SET expires_at = clock_timestamp() - interval '1 second'
 WHERE request_id = $1`, review.Request.RequestID); err != nil {
		t.Fatal(err)
	}
	if _, err := service.DecideFinding(ctx, DecideFindingParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, RequestID: review.Request.RequestID,
		ExpectedRequestRevision: review.Request.Revision, DecisionID: "decision-clock-finding",
		Verdict: VerdictNeedsEvidence, Rationale: "Decided after the database expiry.",
		IdempotencyKey: "decision-clock-finding", RequestDigest: serviceTestDigest("decision-clock-finding"),
	}); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("decide expired finding review with a lagging clock = %v", err)
	}
	expired, err := service.GetReview(ctx, audit.OwnerID, audit.AuditID, review.Request.RequestID)
	if err != nil || expired.State != ReviewExpired {
		t.Fatalf("expired finding review = (%+v, %v)", expired, err)
	}
}
