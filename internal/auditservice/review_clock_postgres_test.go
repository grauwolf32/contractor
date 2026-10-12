package auditservice

import (
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
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
		Verdict: auditstore.VerdictNeedsEvidence, Rationale: "Decided after the database expiry.",
		IdempotencyKey: "decision-clock-finding", RequestDigest: serviceTestDigest("decision-clock-finding"),
	}); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("decide expired finding review with a lagging clock = %v", err)
	}
	expired, err := service.GetReview(ctx, audit.OwnerID, audit.AuditID, review.Request.RequestID)
	if err != nil || expired.State != auditstore.ReviewExpired {
		t.Fatalf("expired finding review = (%+v, %v)", expired, err)
	}
}

func TestReplacingExpiredFindingReviewRecordsExpiryAndRequest(t *testing.T) {
	ctx, pool, service, started, _ := newTimeControlAudit(t, 7200)
	audit := started.Audit
	findingID := seedAuditFinding(t, ctx, pool, audit.ProjectID, audit.OwnerID, audit.AuditID, "replacement")
	params := CreateFindingReviewParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, FindingID: findingID, ExpectedRevision: 1,
		RequestID: "review-old", IdempotencyKey: "review-old", RequestDigest: serviceTestDigest("review-old"),
	}
	old, err := service.CreateFindingReview(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	// Exercise the monotonic timestamp guard even when the stored timestamp
	// is ahead of the database clock. Authority still expires by expires_at.
	var oldUpdated time.Time
	if err := pool.QueryRow(ctx, `
UPDATE audit_review_requests
   SET expires_at=clock_timestamp()-interval '1 second', updated_at='2100-01-01'::timestamptz
 WHERE request_id=$1 RETURNING updated_at`, old.Request.RequestID).Scan(&oldUpdated); err != nil {
		t.Fatal(err)
	}
	store := auditstoretest.NewStore(pool)
	before, err := store.Get(ctx, audit.OwnerID, audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	params.RequestID, params.IdempotencyKey, params.RequestDigest = "review-new", "review-new", serviceTestDigest("review-new")
	fresh, err := service.CreateFindingReview(ctx, params)
	if err != nil || fresh.Replayed || fresh.Request.State != auditstore.ReviewPending {
		t.Fatalf("replacement = (%+v, %v)", fresh, err)
	}
	expired, err := service.GetReview(ctx, audit.OwnerID, audit.AuditID, old.Request.RequestID)
	if err != nil || expired.State != auditstore.ReviewExpired || expired.Revision != old.Request.Revision+1 || !expired.UpdatedAt.After(oldUpdated) {
		t.Fatalf("expired request = (%+v, %v)", expired, err)
	}
	events, err := store.ListEvents(ctx, audit.AuditID, before.EventSequence, 3)
	if err != nil || len(events) != 2 || events[0].Kind != "review.expired" || events[0].EntityID != old.Request.RequestID ||
		events[1].Kind != "review.requested" || events[1].EntityID != fresh.Request.RequestID || events[1].Sequence != events[0].Sequence+1 {
		t.Fatalf("replacement events = (%+v, %v)", events, err)
	}
}

// The owner-wide pending list behind the Inbox offers only requests that can
// still be decided: a stored pending request past its expiry at the
// PostgreSQL clock is left out before anything has expired it, exactly as
// the workspace's pending count leaves it out. The unfiltered list and the
// expired filter keep reporting the stored state.
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
