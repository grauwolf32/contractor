//go:build integration

package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// An analyst's finding decision projects onto audit_findings, whose
// correlation trigger takes the Project correlation fence without waiting
// (FND-R177). Another finding writer of the same Project holds that fence
// until it commits, so the decision aborts with a definite transaction
// conflict. The decision replays its whole transaction, at most three
// transactions, and commits exactly once; other failures are not replayed.
func TestPostgresFindingDecisionReplaysProjectFenceConflict(t *testing.T) {
	t.Run("contention released after the first abort", func(t *testing.T) {
		f := newDecisionRetryFixture(t)
		contender := f.beginConcurrentFindingWriter(t)
		var released error
		trace := &decisionTransactions{}
		trace.afterProjection = func(projection int, err error) {
			if projection == 1 && persistencepostgres.IsTransactionConflict(err) {
				released = contender.Commit(f.ctx)
			}
		}
		service := f.tracedService(t, trace)
		before := f.state(t)
		result, err := f.decide(f.ctx, service, f.request.Revision)
		if err != nil || result.Replayed || released != nil {
			t.Fatalf("decision after released contention = (replayed %t, %v), release %v", result.Replayed, err, released)
		}
		trace.requireAttempts(t, 2, true, false)
		after := f.state(t)
		if want := before.decided(); after != want {
			t.Fatalf("decision state = %+v, want %+v", after, want)
		}
		if result.Finding.State != auditstore.FindingConfirmed || result.Decision.DecisionID != "decision-retry" {
			t.Fatalf("decision result = finding %s, decision %s", result.Finding.State, result.Decision.DecisionID)
		}
		// The stored decision answers its idempotency key without another
		// projection, revision or event.
		replay, err := f.decide(f.ctx, service, f.request.Revision)
		if err != nil || !replay.Replayed || replay.Decision.DecisionID != result.Decision.DecisionID {
			t.Fatalf("decision replay = (replayed %t, decision %s, %v)", replay.Replayed, replay.Decision.DecisionID, err)
		}
		trace.requireAttempts(t, 3, true, false)
		if state := f.state(t); state != after {
			t.Fatalf("decision replay changed state to %+v, want %+v", state, after)
		}
	})

	t.Run("persistent contention", func(t *testing.T) {
		f := newDecisionRetryFixture(t)
		contender := f.beginConcurrentFindingWriter(t)
		trace := &decisionTransactions{}
		service := f.tracedService(t, trace)
		before := f.state(t)
		_, err := f.decide(f.ctx, service, f.request.Revision)
		if !persistencepostgres.IsTransactionConflict(err) {
			t.Fatalf("decision under persistent contention = %v", err)
		}
		trace.requireAttempts(t, 3, true, true, true)
		if state := f.state(t); state != before {
			t.Fatalf("aborted decision changed state to %+v, want %+v", state, before)
		}
		if err := contender.Commit(f.ctx); err != nil {
			t.Fatalf("concurrent finding writer: %v", err)
		}
		// The request stayed pending, so the same request decides it now.
		if _, err := f.decide(f.ctx, service, f.request.Revision); err != nil {
			t.Fatalf("decision after contention ended: %v", err)
		}
		trace.requireAttempts(t, 4, true, true, true, false)
		if state := f.state(t); state != before.decided() {
			t.Fatalf("decision state = %+v, want %+v", state, before.decided())
		}
	})

	t.Run("cancelled after a conflict", func(t *testing.T) {
		f := newDecisionRetryFixture(t)
		f.beginConcurrentFindingWriter(t)
		ctx, cancel := context.WithCancel(f.ctx)
		defer cancel()
		trace := &decisionTransactions{}
		trace.afterProjection = func(int, error) { cancel() }
		service := f.tracedService(t, trace)
		before := f.state(t)
		if _, err := f.decide(ctx, service, f.request.Revision); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled decision = %v", err)
		}
		trace.requireAttempts(t, 1, true)
		if state := f.state(t); state != before {
			t.Fatalf("cancelled decision changed state to %+v, want %+v", state, before)
		}
	})

	t.Run("stale request revision", func(t *testing.T) {
		f := newDecisionRetryFixture(t)
		trace := &decisionTransactions{}
		service := f.tracedService(t, trace)
		before := f.state(t)
		if _, err := f.decide(f.ctx, service, f.request.Revision+1); !errors.Is(err, auditstore.ErrPrecondition) {
			t.Fatalf("stale decision = %v", err)
		}
		trace.requireAttempts(t, 1)
		if state := f.state(t); state != before {
			t.Fatalf("stale decision changed state to %+v, want %+v", state, before)
		}
	})
}

// A child Run that the Audit Controller creates in the same Project holds the
// Project row FOR SHARE until it commits. The finding correlation fence is a
// row of its own (FND-R177), so an analyst finding decision commits in its
// first transaction with one fence step beside it, and the Run creation then
// commits without having waited.
func TestPostgresFindingDecisionSharesTheProjectWithChildRunCreation(t *testing.T) {
	f := newDecisionRetryFixture(t)
	creation := f.beginChildRunCreation(t, "decision-concurrent-run")
	trace := &decisionTransactions{}
	service := f.tracedService(t, trace)
	before := f.state(t)
	result, err := f.decide(f.ctx, service, f.request.Revision)
	if err != nil || result.Replayed {
		t.Fatalf("decision beside a child Run creation = (replayed %t, %v)", result.Replayed, err)
	}
	trace.requireAttempts(t, 1, false)
	if after, want := f.state(t), before.decided(); after != want {
		t.Fatalf("decision state = %+v, want %+v", after, want)
	}
	if err := creation.Commit(f.ctx); err != nil {
		t.Fatalf("commit the child Run creation: %v", err)
	}
}

type decisionRetryFixture struct {
	ctx      context.Context
	pool     *pgxpool.Pool
	findings *findingintake.Service
	finding  string
	request  auditstore.ReviewRequest
}

const (
	decisionRetryOwner   = "decision-retry-owner"
	decisionRetryProject = "decision-retry-project"
	decisionRetryAudit   = "decision-retry-audit"
)

// newDecisionRetryFixture holds one pending finding-triage request of an
// Audit finding in a human-confirmed Audit.
func newDecisionRetryFixture(t *testing.T) decisionRetryFixture {
	t.Helper()
	ctx, pool := collectionTestPool(t)
	createCorrelationAudit(t, ctx, pool, decisionRetryOwner, decisionRetryProject, decisionRetryAudit)
	findings, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	finding := seedAuditFindingDocument(t, ctx, pool, decisionRetryProject, decisionRetryOwner,
		decisionRetryAudit, "decision-retry", auditFindingDocument("decision-retry"))
	service := &Service{pool: pool, findings: findings, now: time.Now}
	pending, err := service.CreateFindingReview(ctx, CreateFindingReviewParams{
		OwnerID: decisionRetryOwner, AuditID: decisionRetryAudit, FindingID: finding, ExpectedRevision: 1,
		RequestID: "review-decision-retry", IdempotencyKey: "review-decision-retry",
		RequestDigest: serviceTestDigest("review-decision-retry"),
	})
	if err != nil {
		t.Fatal(err)
	}
	return decisionRetryFixture{ctx: ctx, pool: pool, findings: findings, finding: finding, request: pending.Request}
}

// tracedService decides on a single-connection pool that trace observes.
func (f decisionRetryFixture) tracedService(t *testing.T, trace *decisionTransactions) *Service {
	t.Helper()
	configuration := f.pool.Config()
	configuration.MaxConns, configuration.MinConns = 1, 0
	configuration.ConnConfig.Tracer = trace
	pool, err := pgxpool.NewWithConfig(f.ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return &Service{pool: pool, findings: f.findings, now: time.Now}
}

func (f decisionRetryFixture) decide(
	ctx context.Context, service *Service, requestRevision uint64,
) (FindingDecisionResult, error) {
	severity := auditstore.SeverityHigh
	return service.DecideFinding(ctx, DecideFindingParams{
		OwnerID: decisionRetryOwner, AuditID: decisionRetryAudit, RequestID: f.request.RequestID,
		ExpectedRequestRevision: requestRevision, DecisionID: "decision-retry",
		Verdict: auditstore.VerdictTruePositive, Severity: &severity, Rationale: "Independent verification.",
		IdempotencyKey: "decision-retry", RequestDigest: serviceTestDigest("decision-retry"),
	})
}

// beginChildRunCreation creates a Run in the fixture Project in an open
// transaction. Its active-Project check holds the Project row FOR SHARE until
// the transaction ends, as an Audit child Run creation does.
func (f decisionRetryFixture) beginChildRunCreation(t *testing.T, runID string) pgx.Tx {
	t.Helper()
	snapshot, _, err := configload.LoadUnionReadOnly("../config/testdata/valid", filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Rollback(context.Background()) })
	project := decisionRetryProject
	if _, err := runstoretest.CreateRun(f.ctx, runstore.NewPostgresStore(tx), runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel, RunID: runID, OwnerID: decisionRetryOwner, ProjectID: &project,
		WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: encoded,
		Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	return tx
}

// beginConcurrentFindingWriter holds the fixture Project's finding
// correlation fence in a transaction that the test ends, as another finding
// writer of the Project does from its fence lock until it commits: an import
// or collection into another Audit takes it through the same auditstore call,
// and a decision on another finding through the finding triggers.
func (f decisionRetryFixture) beginConcurrentFindingWriter(t *testing.T) pgx.Tx {
	t.Helper()
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Rollback(context.Background()) })
	if err := hookstore.NewFindingEventsTx(tx).LockFindingCorrelationFence(f.ctx, decisionRetryProject); err != nil {
		t.Fatal(err)
	}
	return tx
}

// decisionState is what a committed finding decision changes.
type decisionState struct {
	decisions, decidedEvents  int
	requestState              string
	findingRevision           int64
	auditRevision             int64
	projectCorrelationVersion int64
}

func (f decisionRetryFixture) state(t *testing.T) decisionState {
	t.Helper()
	var state decisionState
	if err := f.pool.QueryRow(f.ctx, `
SELECT (SELECT count(*) FROM audit_review_decisions WHERE audit_id = $1),
       (SELECT count(*) FROM audit_events WHERE audit_id = $1 AND kind = 'review.decided'),
       (SELECT state FROM audit_review_requests WHERE audit_id = $1 AND request_id = $2),
       (SELECT revision FROM audit_findings WHERE audit_id = $1 AND finding_id = $3),
       (SELECT revision FROM audits WHERE audit_id = $1),
       (SELECT revision FROM finding_correlation_fences WHERE project_id = $4)`,
		decisionRetryAudit, f.request.RequestID, f.finding, decisionRetryProject,
	).Scan(&state.decisions, &state.decidedEvents, &state.requestState, &state.findingRevision,
		&state.auditRevision, &state.projectCorrelationVersion); err != nil {
		t.Fatal(err)
	}
	return state
}

// decided is s after exactly one committed decision: one decision row and
// review.decided event, a decided request, one finding revision and one
// Project correlation revision, and the review.decided event's Audit revision.
func (s decisionState) decided() decisionState {
	s.decisions++
	s.decidedEvents++
	s.requestState = string(auditstore.ReviewDecided)
	s.findingRevision++
	s.auditRevision++
	s.projectCorrelationVersion++
	return s
}

// decisionTransactions observes a single-connection pool that decides
// findings: each decision attempt, which starts with the Audit's finding
// review advisory lock, and the outcome of each finding projection. Its hook
// runs on the deciding goroutine while the attempt's transaction is open.
type decisionTransactions struct {
	mu              sync.Mutex
	attempts        int
	projections     []error
	afterProjection func(projection int, err error)
}

type decisionProjectionKey struct{}

// decisionReviewLock is the advisory lock that starts each decision attempt
// (LockFindingReviews).
var decisionReviewLock = persistencepostgres.NewAdvisoryKey("audit-finding-review", decisionRetryAudit)

func (c *decisionTransactions) TraceQueryStart(
	ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData,
) context.Context {
	statement := strings.ToLower(strings.TrimSpace(data.SQL))
	c.mu.Lock()
	defer c.mu.Unlock()
	switch {
	case strings.Contains(data.SQL, persistencepostgres.AdvisoryLockSQL("$1")) && len(data.Args) == 1 &&
		fmt.Sprint(data.Args[0]) == string(decisionReviewLock):
		c.attempts++
	case strings.HasPrefix(statement, "update audit_findings"):
		c.projections = append(c.projections, nil)
		return context.WithValue(ctx, decisionProjectionKey{}, len(c.projections))
	}
	return ctx
}

func (c *decisionTransactions) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryEndData) {
	projection, ok := ctx.Value(decisionProjectionKey{}).(int)
	if !ok {
		return
	}
	c.mu.Lock()
	c.projections[projection-1] = data.Err
	hook := c.afterProjection
	c.mu.Unlock()
	if hook != nil {
		hook(projection, data.Err)
	}
}

// requireAttempts checks the decision attempts so far and, for each finding
// projection in order, whether it aborted with a transaction conflict.
func (c *decisionTransactions) requireAttempts(t *testing.T, attempts int, conflicts ...bool) {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.attempts != attempts || len(c.projections) != len(conflicts) {
		t.Fatalf("decision attempts = %d with projections %v, want %d attempts and %d projections",
			c.attempts, c.projections, attempts, len(conflicts))
	}
	for index, conflict := range conflicts {
		err := c.projections[index]
		if persistencepostgres.IsTransactionConflict(err) != conflict || (!conflict && err != nil) {
			t.Fatalf("finding projection %d = %v, want conflict %t", index+1, err, conflict)
		}
	}
}
