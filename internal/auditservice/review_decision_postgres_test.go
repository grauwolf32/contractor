package auditservice

import (
	"context"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// reviewDecisionState renders everything an owner's review decision changes
// in one Audit: its revision and event sequence, its review requests and
// decisions, its findings and its items.
func reviewDecisionState(t *testing.T, ctx context.Context, pool *pgxpool.Pool, auditID string) string {
	t.Helper()
	var state string
	if err := pool.QueryRow(ctx, `
SELECT concat_ws(E'\n',
  (SELECT concat_ws(' ', revision, next_event_sequence) FROM audits WHERE audit_id = $1),
  (SELECT string_agg(concat_ws(' ', request_id, state, revision, updated_at), E'\n' ORDER BY request_id)
     FROM audit_review_requests WHERE audit_id = $1),
  (SELECT string_agg(concat_ws(' ', decision_id, request_id), E'\n' ORDER BY decision_id)
     FROM audit_review_decisions WHERE audit_id = $1),
  (SELECT string_agg(concat_ws(' ', item_id, state, updated_at), E'\n' ORDER BY item_id)
     FROM audit_items WHERE audit_id = $1))`, auditID).Scan(&state); err != nil {
		t.Fatal(err)
	}
	return state
}

// TestReviewDecisionsCommitWithTheirEffectsOrNotAtAll covers V409-005 at the
// service boundary. A finding verdict and an item action decision each commit
// exactly one decision with the decided request, the finding or item
// projection and one review.decided event. When a later step of the same
// transaction fails, none of them remains, and the same request then
// succeeds and replays.
func TestReviewDecisionsCommitWithTheirEffectsOrNotAtAll(t *testing.T) {
	ctx, pool, service, started, _ := newTimeControlAudit(t, 7200)
	audit := started.Audit
	reviews, err := service.ListReviews(ctx, auditstore.ReviewListParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, Limit: 10,
	})
	if err != nil || len(reviews) != 1 || reviews[0].SubjectKind != auditstore.ReviewSubjectItemAction {
		t.Fatalf("item review = (%+v, %v)", reviews, err)
	}
	itemReview := reviews[0]
	itemDecision := DecideActionReviewParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, RequestID: itemReview.RequestID,
		ExpectedRequestRevision: itemReview.Revision, DecisionID: "decision-effects-item",
		Action: auditstore.ReviewApprove, Rationale: "The exact proposed check is approved.",
		IdempotencyKey: "decision-effects-item", RequestDigest: serviceTestDigest("decision-effects-item"),
	}

	// The review.decided event is the last step of both decisions' shared
	// transaction; make it fail after the decision and its projection.
	if _, err := pool.Exec(ctx, `
CREATE FUNCTION fail_review_decided_event() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'injected review.decided failure';
END $$`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
CREATE TRIGGER fail_review_decided_event BEFORE INSERT ON audit_events
   FOR EACH ROW WHEN (NEW.kind = 'review.decided') EXECUTE FUNCTION fail_review_decided_event()`); err != nil {
		t.Fatal(err)
	}
	before := reviewDecisionState(t, ctx, pool, audit.AuditID)
	if _, err := service.DecideActionReview(ctx, itemDecision); err == nil ||
		!strings.Contains(err.Error(), "injected review.decided failure") {
		t.Fatalf("item decision with a failing event = %v", err)
	}
	if after := reviewDecisionState(t, ctx, pool, audit.AuditID); after != before {
		t.Fatalf("failed decisions left changes:\nbefore:\n%s\nafter:\n%s", before, after)
	}
	if _, err := pool.Exec(ctx, `DROP TRIGGER fail_review_decided_event ON audit_events`); err != nil {
		t.Fatal(err)
	}

	item, err := service.DecideActionReview(ctx, itemDecision)
	if err != nil || item.Replayed || item.Request.State != auditstore.ReviewDecided ||
		item.Request.Revision != itemReview.Revision+1 || item.Decision.DecisionID != itemDecision.DecisionID {
		t.Fatalf("item decision = (%+v, %v)", item, err)
	}
	var itemState string
	if err := pool.QueryRow(ctx, `
SELECT item.state FROM audit_items AS item
  JOIN audit_review_requests AS request
    ON request.audit_id = item.audit_id AND request.subject_id = item.item_id
 WHERE request.request_id = $1`, itemReview.RequestID).Scan(&itemState); err != nil ||
		itemState != string(auditstore.ItemReady) {
		t.Fatalf("approved item state = (%s, %v)", itemState, err)
	}

	// A replay returns the stored decisions and writes nothing more.
	settled := reviewDecisionState(t, ctx, pool, audit.AuditID)
	if replay, err := service.DecideActionReview(ctx, itemDecision); err != nil || !replay.Replayed ||
		replay.Decision.DecisionID != itemDecision.DecisionID {
		t.Fatalf("item decision replay = (%+v, %v)", replay, err)
	}
	if after := reviewDecisionState(t, ctx, pool, audit.AuditID); after != settled {
		t.Fatalf("replays changed the Audit:\nbefore:\n%s\nafter:\n%s", settled, after)
	}

	// Each request has exactly one decision and one review.decided event.
	for _, decision := range []struct{ requestID, decisionID string }{
		{itemDecision.RequestID, itemDecision.DecisionID},
	} {
		var decisions, events int
		if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM audit_review_decisions WHERE request_id = $1),
       (SELECT count(*) FROM audit_events
         WHERE audit_id = $3 AND kind = 'review.decided' AND entity_id = $2)`,
			decision.requestID, decision.decisionID, audit.AuditID).Scan(&decisions, &events); err != nil ||
			decisions != 1 || events != 1 {
			t.Fatalf("request %s: decisions=%d review.decided events=%d (%v)",
				decision.requestID, decisions, events, err)
		}
	}
}
