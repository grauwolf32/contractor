package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// reviewDecisionFixture contains two pending journal requests and a second
// Audit of the same owner and Project.
type reviewDecisionFixture struct {
	pool                           *pgxpool.Pool
	ownerID, auditID, otherAuditID string
	requests                       []string
}

func newReviewDecisionFixture(t *testing.T, ctx context.Context, suffix string) reviewDecisionFixture {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-decision-" + suffix, OwnerID: "owner-decision-" + suffix,
		Kind: projectstore.KindProject, Name: "Review decisions",
		IdempotencyKey: "project", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := newTestStore(pool)
	draft := func(auditID, key string) auditstore.Audit {
		audit, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
			AuditID: auditID, OwnerID: project.OwnerID, ProjectID: project.ProjectID,
			Profile:         auditstore.ProfileIdentity{Name: "checklist", Version: "1", Digest: testDigest("2")},
			ProfileSnapshot: json.RawMessage(`{"name":"checklist"}`),
			InputSelection:  json.RawMessage(`{}`),
			Limits: auditstore.Limits{
				MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 2, MaxItemsTotal: 2,
				MaxSubmittedRuns: 2, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
			},
			IdempotencyKey: key, RequestDigest: testDigest("3"),
		})
		if err != nil {
			t.Fatal(err)
		}
		return audit
	}
	audit := draft("audit-decision-"+suffix, "audit")
	other := draft("audit-decision-other-"+suffix, "audit-other")
	seedReviewRows(t, ctx, pool, audit.AuditID, "item-one", "item-two")
	// These fixed request rows exercise the surviving review journal writer;
	// they do not recreate retired Round admission or execution.
	fixture := reviewDecisionFixture{
		pool: pool, ownerID: project.OwnerID, auditID: audit.AuditID, otherAuditID: other.AuditID,
	}
	for _, subject := range []string{"item-one", "item-two"} {
		requestID := "review-" + subject
		if _, err := pool.Exec(ctx, `INSERT INTO audit_review_requests
(request_id,audit_id,kind,subject_revision,subject_digest,requested_actions,
 idempotency_key,request_digest,subject_kind,subject_id)
VALUES($1,$2,'active-check-approval',1,$3,'["approve","reject"]',$1,$3,'audit-item-action',$4)`,
			requestID, audit.AuditID, testDigest("4"), subject); err != nil {
			t.Fatal(err)
		}
		fixture.requests = append(fixture.requests, requestID)
	}
	return fixture
}

// decision is an item-action approval of request on auditID that differs
// from every other decision by suffix. It pins the request's subject, or a
// placeholder subject when the request does not exist.
func (f reviewDecisionFixture) decision(
	t *testing.T, ctx context.Context, auditID, requestID, suffix string,
) auditstore.ReviewDecisionWriteParams {
	t.Helper()
	var revision uint64
	var digest string
	err := f.pool.QueryRow(ctx, `
SELECT subject_revision, subject_digest FROM audit_review_requests WHERE request_id = $1`,
		requestID).Scan(&revision, &digest)
	if errors.Is(err, pgx.ErrNoRows) {
		revision, digest = 1, testDigest("7")
	} else if err != nil {
		t.Fatal(err)
	}
	return auditstore.ReviewDecisionWriteParams{
		DecisionID: "decision-" + suffix, RequestID: requestID, AuditID: auditID,
		ActorID: f.ownerID, Action: string(auditstore.ReviewApprove),
		Rationale:       "The exact proposed action is approved.",
		SubjectRevision: revision, SubjectDigest: digest,
		IdempotencyKey: "decision-" + suffix, RequestDigest: testDigest("8"),
	}
}

// record runs the store write the way DecideActionReview does: in one
// transaction under the Audit's action-review lock and, when lockAuditID is
// set, with request p.RequestID of that Audit and the Audit row locked. then
// runs in the same transaction after a successful write.
func (f reviewDecisionFixture) record(
	ctx context.Context, lockAuditID string, p auditstore.ReviewDecisionWriteParams,
	then func(pgx.Tx) error,
) error {
	return persistencepostgres.InTx(ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		reviews := NewReviewTxStore(tx, artifactpgtest.NewStorage())
		if err := reviews.LockActionReviews(ctx, p.AuditID); err != nil {
			return err
		}
		if lockAuditID != "" {
			if _, err := reviews.LockActionReview(ctx, f.ownerID, lockAuditID, p.RequestID); err != nil {
				return err
			}
		}
		if err := reviews.RecordReviewDecision(ctx, p); err != nil {
			return err
		}
		if then != nil {
			return then(tx)
		}
		return nil
	})
}

// ledger renders every stored review request and decision, so a refused
// write can be shown to have changed nothing.
func (f reviewDecisionFixture) ledger(t *testing.T, ctx context.Context) string {
	t.Helper()
	var ledger strings.Builder
	rows, err := f.pool.Query(ctx, `
SELECT request_id, audit_id, state, revision, updated_at
  FROM audit_review_requests ORDER BY request_id`)
	if err != nil {
		t.Fatal(err)
	}
	for rows.Next() {
		var requestID, auditID, state string
		var revision int64
		var updatedAt time.Time
		if err := rows.Scan(&requestID, &auditID, &state, &revision, &updatedAt); err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(&ledger, "request %s %s %s %d %s\n", requestID, auditID, state, revision,
			updatedAt.UTC().Format(time.RFC3339Nano))
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	rows, err = f.pool.Query(ctx, `
SELECT decision_id, request_id, audit_id, actor_id, action, rationale,
       idempotency_key, request_digest, created_at
  FROM audit_review_decisions ORDER BY decision_id`)
	if err != nil {
		t.Fatal(err)
	}
	for rows.Next() {
		var decisionID, requestID, auditID, actorID, action, rationale, key, digest string
		var createdAt time.Time
		if err := rows.Scan(&decisionID, &requestID, &auditID, &actorID, &action, &rationale,
			&key, &digest, &createdAt); err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(&ledger, "decision %s %s %s %s %s %q %s %s %s\n", decisionID, requestID, auditID,
			actorID, action, rationale, key, digest, createdAt.UTC().Format(time.RFC3339Nano))
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	return ledger.String()
}

func (f reviewDecisionFixture) request(t *testing.T, ctx context.Context, requestID string) (string, int64) {
	t.Helper()
	var state string
	var revision int64
	if err := f.pool.QueryRow(ctx, `
SELECT state, revision FROM audit_review_requests WHERE request_id = $1`,
		requestID).Scan(&state, &revision); err != nil {
		t.Fatal(err)
	}
	return state, revision
}

func (f reviewDecisionFixture) decisions(t *testing.T, ctx context.Context, requestID string) int {
	t.Helper()
	var count int
	if err := f.pool.QueryRow(ctx, `
SELECT count(*) FROM audit_review_decisions WHERE request_id = $1`, requestID).Scan(&count); err != nil {
		t.Fatal(err)
	}
	return count
}

// TestPostgresReviewDecisionRequiresItsPendingRequest covers V409-005: the
// store appends a decision only together with the decided transition of its
// own pending request. An expired, decided, absent or other Audit's request
// is a precondition failure that writes nothing, even when the caller holds
// the request lock, and a surrounding transaction failure undoes a decision.
func TestPostgresReviewDecisionRequiresItsPendingRequest(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	f := newReviewDecisionFixture(t, ctx, "gate")
	decided, expired := f.requests[0], f.requests[1]
	_, pendingRevision := f.request(t, ctx, decided)

	refused := func(name, lockAuditID string, p auditstore.ReviewDecisionWriteParams) {
		t.Helper()
		before := f.ledger(t, ctx)
		if err := f.record(ctx, lockAuditID, p, nil); !errors.Is(err, auditstore.ErrPrecondition) {
			t.Errorf("%s decision = %v, want ErrPrecondition", name, err)
		}
		if after := f.ledger(t, ctx); after != before {
			t.Errorf("%s decision changed the review ledger:\nbefore:\n%safter:\n%s", name, before, after)
		}
	}

	// A request that does not exist, or that belongs to another Audit, is
	// never decided through this one.
	refused("absent request", "", f.decision(t, ctx, f.auditID, "request-absent", "absent"))
	refused("Audit mismatch", f.auditID, f.decision(t, ctx, f.otherAuditID, decided, "mismatch"))

	// A failure later in the caller's transaction undoes the decision and the
	// decided request with it.
	rollback := errors.New("later step failed")
	err := f.record(ctx, f.auditID, f.decision(t, ctx, f.auditID, decided, "rolled-back"), func(tx pgx.Tx) error {
		var state string
		var decisions int
		if err := tx.QueryRow(ctx, `
SELECT request.state, (SELECT count(*) FROM audit_review_decisions WHERE request_id = request.request_id)
  FROM audit_review_requests AS request WHERE request.request_id = $1`, decided).Scan(&state, &decisions); err != nil {
			return err
		}
		if state != string(auditstore.ReviewDecided) || decisions != 1 {
			return fmt.Errorf("inside the transaction: state=%s decisions=%d", state, decisions)
		}
		return rollback
	})
	if !errors.Is(err, rollback) {
		t.Fatalf("rolled-back decision = %v", err)
	}
	if state, revision := f.request(t, ctx, decided); state != string(auditstore.ReviewPending) ||
		revision != pendingRevision || f.decisions(t, ctx, decided) != 0 {
		t.Fatalf("after rollback: state=%s revision=%d decisions=%d", state, revision, f.decisions(t, ctx, decided))
	}

	// The pending request is decided with exactly one decision.
	if err := f.record(ctx, f.auditID, f.decision(t, ctx, f.auditID, decided, "accepted"), nil); err != nil {
		t.Fatalf("pending request decision: %v", err)
	}
	if state, revision := f.request(t, ctx, decided); state != string(auditstore.ReviewDecided) ||
		revision != pendingRevision+1 || f.decisions(t, ctx, decided) != 1 {
		t.Fatalf("decided request: state=%s revision=%d decisions=%d", state, revision, f.decisions(t, ctx, decided))
	}

	// A decided request keeps its one decision.
	refused("decided request", f.auditID, f.decision(t, ctx, f.auditID, decided, "second"))

	// A request the database clock expired under its lock is not decided by
	// a write that follows under the same lock contract.
	if _, err := f.pool.Exec(ctx, `
UPDATE audit_review_requests SET expires_at = clock_timestamp() - interval '1 second'
 WHERE request_id = $1`, expired); err != nil {
		t.Fatal(err)
	}
	if err := persistencepostgres.InTx(ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		reviews := NewReviewTxStore(tx, artifactpgtest.NewStorage())
		if _, err := reviews.LockActionReview(ctx, f.ownerID, f.auditID, expired); err != nil {
			return err
		}
		changed, err := reviews.ExpireReviewAtDatabaseTime(ctx, expired)
		if err == nil && !changed {
			err = errors.New("request did not expire")
		}
		if err != nil {
			return err
		}
		return reviews.AppendReviewEvent(ctx, auditstore.ReviewEventParams{
			AuditID: f.auditID, Kind: "review.expired", EntityID: expired,
			Summary: map[string]any{"subjectKind": "audit-item-action", "subjectId": "item-two"},
		})
	}); err != nil {
		t.Fatalf("expire request: %v", err)
	}
	if state, _ := f.request(t, ctx, expired); state != string(auditstore.ReviewExpired) {
		t.Fatalf("expired request state = %s", state)
	}
	refused("expired request", f.auditID, f.decision(t, ctx, f.auditID, expired, "expired"))
}

// TestPostgresReviewDecisionCheckViolationIsInvalid covers V368-023: a review
// decision the stored constraints refuse is ErrInvalid, never an
// unclassified storage error, and the request it would have decided stays
// pending. The request exists and is pending, so the constraint is reached.
func TestPostgresReviewDecisionCheckViolationIsInvalid(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	f := newReviewDecisionFixture(t, ctx, "check")
	p := f.decision(t, ctx, f.auditID, f.requests[0], "invalid-key")
	p.IdempotencyKey = "key with spaces"
	before := f.ledger(t, ctx)
	if err := f.record(ctx, f.auditID, p, nil); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("review decision with an invalid key = %v, want ErrInvalid", err)
	}
	if after := f.ledger(t, ctx); after != before {
		t.Fatalf("refused decision changed the review ledger:\nbefore:\n%safter:\n%s", before, after)
	}
}
