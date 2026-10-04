package auditservice

import (
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// TestAuditResumeRecordsRenewedReviewEvents resumes an Audit whose item
// review expired while paused. The renewal records its events and revisions
// before audit.resumed, and only the fresh request can authorize the item.
func TestAuditResumeRecordsRenewedReviewEvents(t *testing.T) {
	ctx, pool, service, started, now := newTimeControlAudit(t, 7200)
	paused, err := service.Pause(ctx, MutationParams{
		OwnerID: started.Audit.OwnerID, AuditID: started.Audit.AuditID, ExpectedRevision: started.Audit.Revision,
		IdempotencyKey: "pause-renewal", RequestDigest: serviceTestDigest("pause-renewal"),
	})
	if err != nil {
		t.Fatal(err)
	}
	reviews, err := service.ListReviews(ctx, ReviewListParams{
		OwnerID: started.Audit.OwnerID, AuditID: started.Audit.AuditID, Limit: 10,
	})
	if err != nil || len(reviews) != 1 || reviews[0].State != ReviewPending {
		t.Fatalf("paused reviews = (%+v, %v)", reviews, err)
	}
	expired := reviews[0]
	if _, err := pool.Exec(ctx, `UPDATE audit_review_requests
SET expires_at=clock_timestamp()-interval '1 minute' WHERE request_id=$1`, expired.RequestID); err != nil {
		t.Fatal(err)
	}
	*now = paused.Audit.PausedAt.Add(time.Hour)
	params := MutationParams{
		OwnerID: paused.Audit.OwnerID, AuditID: paused.Audit.AuditID, ExpectedRevision: paused.Audit.Revision,
		IdempotencyKey: "resume-renewal", RequestDigest: serviceTestDigest("resume-renewal"),
	}
	resumed, err := service.Resume(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if resumed.Audit.State != auditstore.AuditActive || resumed.Audit.Revision != paused.Audit.Revision+3 ||
		resumed.Audit.EventSequence != paused.Audit.EventSequence+3 || resumed.Audit.DeadlineAt == nil {
		t.Fatalf("resumed Audit = (%s, revision %d, sequence %d, deadline %v), paused at revision %d",
			resumed.Audit.State, resumed.Audit.Revision, resumed.Audit.EventSequence, resumed.Audit.DeadlineAt,
			paused.Audit.Revision)
	}
	reviews, err = service.ListReviews(ctx, ReviewListParams{
		OwnerID: started.Audit.OwnerID, AuditID: started.Audit.AuditID, Limit: 10,
	})
	if err != nil || len(reviews) != 2 {
		t.Fatalf("renewed reviews = (%+v, %v)", reviews, err)
	}
	var renewed, stale ReviewRequest
	for _, review := range reviews {
		if review.RequestID == expired.RequestID {
			stale = review
		} else {
			renewed = review
		}
	}
	if stale.State != ReviewExpired || stale.Revision != expired.Revision+1 {
		t.Fatalf("expired review = %+v", stale)
	}
	if renewed.State != ReviewPending || renewed.SubjectID != expired.SubjectID || renewed.ExpiresAt == nil ||
		!renewed.ExpiresAt.Equal(*resumed.Audit.DeadlineAt) {
		t.Fatalf("renewed review = %+v, deadline %v", renewed, resumed.Audit.DeadlineAt)
	}
	events, err := auditstore.NewPostgresStore(pool).ListEvents(ctx, paused.Audit.AuditID, paused.Audit.EventSequence, 10)
	if err != nil || len(events) != 3 {
		t.Fatalf("Resume events = (%+v, %v)", events, err)
	}
	for index, want := range []struct {
		kind, entity string
		revision     uint64
	}{
		{"review.expired", expired.RequestID, expired.Revision + 1},
		{"review.requested", renewed.RequestID, 1},
		{"audit.resumed", paused.Audit.AuditID, resumed.Audit.Revision},
	} {
		event := events[index]
		if event.Sequence != paused.Audit.EventSequence+uint64(index)+1 || event.Kind != want.kind ||
			event.EntityID != want.entity || event.EntityRevision == nil || *event.EntityRevision != want.revision {
			t.Fatalf("Resume event %d = %+v, want %+v", index, event, want)
		}
	}
	replayed, err := service.Resume(ctx, params)
	if err != nil || !replayed.Replayed || replayed.Audit.Revision != resumed.Audit.Revision {
		t.Fatalf("Resume replay = (%+v, %v)", replayed, err)
	}

	decide := func(request ReviewRequest, key string) error {
		_, err := service.DecideActionReview(ctx, DecideActionReviewParams{
			OwnerID: started.Audit.OwnerID, AuditID: started.Audit.AuditID, RequestID: request.RequestID,
			ExpectedRequestRevision: request.Revision, DecisionID: key, Action: ReviewApprove,
			Rationale: "Approve the exact task.", IdempotencyKey: key, RequestDigest: serviceTestDigest(key),
		})
		return err
	}
	if err := decide(stale, "late-decision"); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("decision on the expired review = %v, want precondition", err)
	}
	if err := decide(renewed, "renewed-decision"); err != nil {
		t.Fatalf("decision on the renewed review: %v", err)
	}
	var state string
	if err := pool.QueryRow(ctx, `SELECT state FROM audit_items WHERE audit_id=$1 AND item_id=$2`,
		started.Audit.AuditID, renewed.SubjectID).Scan(&state); err != nil || state != string(auditstore.ItemReady) {
		t.Fatalf("approved item state = (%s, %v)", state, err)
	}
}
