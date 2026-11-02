package auditservice

import (
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
	reviews, err := service.ListReviews(ctx, auditstore.ReviewListParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, Limit: 10,
	})
	if err != nil || len(reviews) != 1 || reviews[0].SubjectKind != auditstore.ReviewSubjectItemAction ||
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
		Action: auditstore.ReviewApprove, Rationale: "Approved before the database expiry.",
		IdempotencyKey: "decision-clock-item", RequestDigest: serviceTestDigest("decision-clock-item"),
	}); err != nil {
		t.Fatalf("decide unexpired item review with a skewed clock: %v", err)
	}

}

func TestOwnerPendingReviewsLeaveOutRequestsPastExpiry(t *testing.T) {
	ctx, pool, service, started, now := newTimeControlAudit(t, 7200)
	audit := started.Audit
	reviews, err := service.ListReviews(ctx, auditstore.ReviewListParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, Limit: 10,
	})
	if err != nil || len(reviews) != 1 || reviews[0].State != auditstore.ReviewPending {
		t.Fatalf("item review = (%+v, %v)", reviews, err)
	}
	requestID := reviews[0].RequestID
	listed := func(state *auditstore.ReviewState) []string {
		t.Helper()
		requests, err := service.ListOwnerReviews(ctx, auditstore.OwnerReviewListParams{
			OwnerID: audit.OwnerID, State: state, Limit: 50,
		})
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]string, 0, len(requests))
		for _, request := range requests {
			ids = append(ids, request.RequestID)
		}
		return ids
	}
	pending, expired := auditstore.ReviewPending, auditstore.ReviewExpired
	setExpiry := func(expiry string) {
		t.Helper()
		if _, err := pool.Exec(ctx, `UPDATE audit_review_requests SET expires_at = `+expiry+`
 WHERE request_id = $1`, requestID); err != nil {
			t.Fatal(err)
		}
	}
	workspacePending := func() int {
		t.Helper()
		workspace, err := service.GetWorkspace(ctx, audit.OwnerID, audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		return workspace.PendingReviews
	}
	if got := listed(&pending); len(got) != 1 || got[0] != requestID || workspacePending() != 1 {
		t.Fatalf("unexpired pending list = %v, workspace = %d", got, workspacePending())
	}

	// A process clock past the expiry does not hide a request the database
	// clock still allows deciding.
	*now = now.Add(30 * 24 * time.Hour)
	setExpiry(`clock_timestamp() + interval '1 hour'`)
	if got := listed(&pending); len(got) != 1 || got[0] != requestID {
		t.Fatalf("pending list with a skewed process clock = %v", got)
	}

	setExpiry(`clock_timestamp() - interval '1 second'`)
	if got := listed(&pending); len(got) != 0 || workspacePending() != 0 {
		t.Fatalf("pending list after the expiry = %v, workspace = %d", got, workspacePending())
	}
	if got := listed(nil); len(got) != 1 || got[0] != requestID {
		t.Fatalf("unfiltered list after the expiry = %v", got)
	}
	if got := listed(&expired); len(got) != 0 {
		t.Fatalf("expired list before the expiry is stored = %v", got)
	}
	var state string
	if err := pool.QueryRow(ctx, `SELECT state FROM audit_review_requests WHERE request_id = $1`,
		requestID).Scan(&state); err != nil || state != string(auditstore.ReviewPending) {
		t.Fatalf("listing changed the stored request: state=%q err=%v", state, err)
	}
}
