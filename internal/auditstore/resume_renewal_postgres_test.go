package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
)

// TestPostgresResumeRenewalRecordsReviewEvents covers every renewable item
// review state: an expired pending request, an expired approval and an
// already expired request. A live request stays untouched.
func TestPostgresResumeRenewalRecordsReviewEvents(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-resume-renewal", OwnerID: "owner-resume-renewal",
		Kind: projectstore.KindProject, Name: "Resume renewal",
		IdempotencyKey: "project", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := NewPostgresStore(pool)
	audit, _, err := store.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-resume-renewal", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         ProfileIdentity{Name: "checklist", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"checklist"}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 4, MaxItemsTotal: 4,
			MaxSubmittedRuns: 4, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
		},
		IdempotencyKey: "audit", RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	items := make([]MaterializedItem, 0, 4)
	for index, approval := range []ItemApprovalKind{
		ItemApprovalActiveCheck, ItemApprovalApplicability, ItemApprovalActiveCheck, ItemApprovalActiveCheck,
	} {
		key := []string{"one", "two", "three", "four"}[index]
		items = append(items, MaterializedItem{
			ItemID: "item-" + key, ItemKey: "check-" + key, Ordinal: index, Kind: "checklist",
			SubjectKey: "subject-" + key, Task: testExact("audits", "task-"+key, "r1"),
			Origin: testOrigin("check-" + key), WorkflowRole: "check",
			InitialState: ItemAwaitingReview, ApprovalKind: approval,
			ApprovalDigest: testDigest(string(rune('4' + index))), Coverage: emptyCoverage(),
		})
	}
	audit, _, err = store.MaterializeRound(ctx, MaterializeRoundParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: "round-resume-renewal", RoundOrdinal: 1,
		Manifest:         testExact("audits", "worklist", "r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":[],"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), Items: items,
		IdempotencyKey: "start", RequestDigest: testDigest("8"),
	})
	if err != nil {
		t.Fatal(err)
	}
	audit, _, err = store.Transition(ctx, TransitionParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		ExpectedState: AuditActive, TargetState: AuditPaused,
		IdempotencyKey: "pause", RequestDigest: testDigest("9"),
	})
	if err != nil {
		t.Fatal(err)
	}
	// Item two was approved and item four's request already expired; the
	// paused clock then outlives every request except item three's.
	for _, statement := range []string{
		`INSERT INTO audit_review_decisions (
    decision_id, request_id, audit_id, finding_id, actor_id, action, verdict, severity,
    rationale, duplicate_target_id, subject_revision, subject_digest, idempotency_key, request_digest
) SELECT 'decision-two', request_id, audit_id, NULL, 'owner-resume-renewal', 'approve', NULL, NULL,
         'Approve the exact task.', NULL, 1, subject_digest, 'decision-two', subject_digest
    FROM audit_review_requests WHERE request_id = 'review-item-two'`,
		`UPDATE audit_review_requests SET state = 'decided', revision = 2 WHERE request_id = 'review-item-two'`,
		`UPDATE audit_items SET state = 'ready' WHERE item_id = 'item-two'`,
		`UPDATE audit_review_requests SET state = 'expired', revision = 2 WHERE request_id = 'review-item-four'`,
		`UPDATE audit_review_requests SET expires_at = clock_timestamp() - interval '1 minute'
          WHERE request_id IN ('review-item-one', 'review-item-two', 'review-item-four')`,
	} {
		if _, err := pool.Exec(ctx, statement); err != nil {
			t.Fatal(err)
		}
	}
	deadline := time.Now().Add(2 * time.Hour).UTC().Truncate(time.Microsecond)
	params := ResumeRenewalParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision, DeadlineAt: &deadline,
	}
	if _, err := store.RenewItemReviewsForResume(ctx, params); err == nil {
		t.Fatal("renewal ran outside the Resume transaction")
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	txStore := NewPostgresStore(tx)
	stale := params
	stale.ExpectedRevision--
	if _, err := txStore.RenewItemReviewsForResume(ctx, stale); !errors.Is(err, ErrPrecondition) {
		t.Fatalf("stale renewal = %v, want precondition", err)
	}
	revision, err := txStore.RenewItemReviewsForResume(ctx, params)
	if err != nil || revision != audit.Revision+5 {
		t.Fatalf("renewal = (%d, %v), want revision %d", revision, err, audit.Revision+5)
	}
	params.ExpectedRevision = revision
	if repeated, err := txStore.RenewItemReviewsForResume(ctx, params); err != nil || repeated != revision {
		t.Fatalf("repeated renewal = (%d, %v), want unchanged revision %d", repeated, err, revision)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	renewed, err := store.Get(ctx, audit.OwnerID, audit.AuditID)
	if err != nil || renewed.Revision != revision || renewed.EventSequence != audit.EventSequence+5 ||
		renewed.Revision != renewed.EventSequence || renewed.State != AuditPaused {
		t.Fatalf("renewed Audit = (%+v, %v)", renewed, err)
	}
	fresh := make(map[string]string)
	rows, err := pool.Query(ctx, `
SELECT request_id, subject_id, state, revision, expires_at, idempotency_key
  FROM audit_review_requests WHERE audit_id = $1 ORDER BY request_id`, audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	for rows.Next() {
		var requestID, subjectID, state, idempotencyKey string
		var requestRevision int64
		var expiresAt *time.Time
		if err := rows.Scan(&requestID, &subjectID, &state, &requestRevision, &expiresAt, &idempotencyKey); err != nil {
			t.Fatal(err)
		}
		want := map[string]struct {
			state    string
			revision int64
		}{
			"review-item-one": {"expired", 2}, "review-item-two": {"expired", 3},
			"review-item-three": {"pending", 1}, "review-item-four": {"expired", 2},
		}
		if expected, original := want[requestID]; original {
			if state != expected.state || requestRevision != expected.revision {
				t.Errorf("request %s = (%s, %d), want (%s, %d)", requestID, state, requestRevision,
					expected.state, expected.revision)
			}
			continue
		}
		if state != "pending" || requestRevision != 1 || expiresAt == nil || !expiresAt.Equal(deadline) ||
			idempotencyKey != requestID || fresh[subjectID] != "" {
			t.Errorf("fresh request %s for %s = (%s, %d, %v)", requestID, subjectID, state, requestRevision, expiresAt)
		}
		fresh[subjectID] = requestID
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(fresh) != 3 || fresh["item-one"] == "" || fresh["item-two"] == "" || fresh["item-four"] == "" {
		t.Fatalf("fresh requests = %+v", fresh)
	}
	storedItems, err := store.ListItems(ctx, audit.AuditID)
	if err != nil || len(storedItems) != 4 {
		t.Fatalf("items = (%+v, %v)", storedItems, err)
	}
	for _, item := range storedItems {
		if item.State != ItemAwaitingReview {
			t.Errorf("item %s state = %s, want awaiting review", item.ItemID, item.State)
		}
	}
	events, err := store.ListEvents(ctx, audit.AuditID, audit.EventSequence, MaxPageSize)
	if err != nil {
		t.Fatal(err)
	}
	type wantEvent struct {
		kind, entity string
		revision     uint64
		subject      string
	}
	want := []wantEvent{
		{"review.expired", "review-item-one", 2, "item-one"},
		{"review.requested", fresh["item-one"], 1, "item-one"},
		{"review.expired", "review-item-two", 3, "item-two"},
		{"review.requested", fresh["item-two"], 1, "item-two"},
		{"review.requested", fresh["item-four"], 1, "item-four"},
	}
	if len(events) != len(want) {
		t.Fatalf("renewal events = %+v", events)
	}
	for index, event := range events {
		var summary struct {
			SubjectKind string `json:"subjectKind"`
			SubjectID   string `json:"subjectId"`
			Kind        string `json:"kind"`
		}
		if err := json.Unmarshal(event.Summary, &summary); err != nil {
			t.Fatal(err)
		}
		expected := want[index]
		if event.Sequence != audit.EventSequence+uint64(index)+1 || event.Kind != expected.kind ||
			event.EntityID != expected.entity || event.EntityRevision == nil || *event.EntityRevision != expected.revision ||
			summary.SubjectKind != "audit-item-action" || summary.SubjectID != expected.subject ||
			summary.Kind != string(items[map[string]int{"item-one": 0, "item-two": 1, "item-four": 3}[expected.subject]].ApprovalKind) {
			t.Errorf("event %d = %+v (%+v), want %+v", index, event, summary, expected)
		}
	}
}
