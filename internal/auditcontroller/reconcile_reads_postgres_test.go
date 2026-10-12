//go:build integration

package auditcontroller

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// reconcileQueryTrace counts every statement a traced Controller issues and,
// separately, those whose text contains one of the watched fragments.
type reconcileQueryTrace struct {
	mu      sync.Mutex
	total   int
	watched map[string]int
}

func newReconcileQueryTrace(fragments ...string) *reconcileQueryTrace {
	trace := &reconcileQueryTrace{watched: make(map[string]int, len(fragments))}
	for _, fragment := range fragments {
		trace.watched[fragment] = 0
	}
	return trace
}

func (q *reconcileQueryTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.total++
	for fragment := range q.watched {
		if strings.Contains(data.SQL, fragment) {
			q.watched[fragment]++
		}
	}
	return ctx
}

func (*reconcileQueryTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// take returns the total and the watched counts since the last take.
func (q *reconcileQueryTrace) take() (int, map[string]int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	watched := make(map[string]int, len(q.watched))
	for fragment, count := range q.watched {
		watched[fragment] = count
		q.watched[fragment] = 0
	}
	total := q.total
	q.total = 0
	return total, watched
}

// tracedController reconciles the harness Audit with a Store and RunStore
// that read through their own pool on the harness schema, so trace sees
// every statement of a Controller pass. Run creation and submission
// preparation keep the harness pool.
func (h *postgresControllerHarness) tracedController(
	t *testing.T, ctx context.Context, holderID string, trace pgx.QueryTracer,
) *Controller {
	t.Helper()
	configuration := h.pool.Config()
	configuration.ConnConfig.Tracer = trace
	pool, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	var ids int
	controller, err := New(
		auditstoretest.NewStore(pool), auditstoretest.NewStore(pool), runstore.NewPostgresStore(pool), rundeletion.New(pool),
		h.runService, h.builder(t), &postgresNotifier{},
		Options{
			HolderID: holderID, ClaimLease: 5 * time.Second,
			OperationTimeout: 2 * time.Second, ClaimBatch: 1,
			NewID: func(prefix string) (string, error) {
				ids++
				return fmt.Sprintf("%s%s_%d", prefix, holderID, ids), nil
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return controller
}

// runStatement is the shape of a single Run read by identity, which the
// Controller made once per submitted execution before Run states joined the
// reconcile snapshot.
const runStatement = "FROM workflow_runs WHERE run_id = $1"

// TestPostgresControllerFullWindowPassCostIsConstantInChildRuns reconciles an
// Audit whose dispatch window is full, first with two and then with six
// submitted child Runs, and finds the same statements both times: the
// snapshot carries every child Run's state, so no Run is read per execution.
func TestPostgresControllerFullWindowPassCostIsConstantInChildRuns(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 6)
	settings := settingsstore.NewPostgresStore(harness.pool)
	if _, err := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 2, ExpectedRevision: 1,
	}); err != nil {
		t.Fatal(err)
	}
	dispatcher := harness.controller(t)
	trace := newReconcileQueryTrace(runStatement)
	observer := harness.tracedController(t, ctx, "traced-observer", trace)

	measure := func(wantRuns int) int {
		t.Helper()
		trace.take()
		if worked, err := observer.RunOnce(ctx); err != nil || worked {
			t.Fatalf("full-window pass with %d child Runs = (%t, %v), want no change", wantRuns, worked, err)
		}
		total, watched := trace.take()
		if watched[runStatement] != 0 {
			t.Fatalf("full-window pass with %d child Runs read %d Runs one by one", wantRuns, watched[runStatement])
		}
		executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
		if err != nil || len(executions) != wantRuns {
			t.Fatalf("executions = (%d, %v), want %d", len(executions), err, wantRuns)
		}
		for _, execution := range executions {
			if execution.State != auditstore.ExecutionSubmitted || execution.RunID == nil {
				t.Fatalf("execution %s is %s, want submitted", execution.ExecutionID, execution.State)
			}
		}
		return total
	}

	// Activate the Round, then fill the window of two.
	for step, wantWorked := range []bool{true, true, true, false} {
		if worked, err := dispatcher.RunOnce(ctx); err != nil || worked != wantWorked {
			t.Fatalf("fill window step %d = (%t, %v), want worked=%t", step, worked, err, wantWorked)
		}
	}
	two := measure(2)

	if _, err := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 6, ExpectedRevision: 2,
	}); err != nil {
		t.Fatal(err)
	}
	for step, wantWorked := range []bool{true, true, true, true, false} {
		if worked, err := dispatcher.RunOnce(ctx); err != nil || worked != wantWorked {
			t.Fatalf("widen window step %d = (%t, %v), want worked=%t", step, worked, err, wantWorked)
		}
	}
	six := measure(6)
	if two != six {
		t.Fatalf("full-window pass issued %d statements with 2 child Runs and %d with 6", two, six)
	}
}

// TestPostgresControllerDispatchReadsBatchAttemptsOnce dispatches one batch of
// four ready items and reads their next attempts in one statement, not one
// per member.
func TestPostgresControllerDispatchReadsBatchAttemptsOnce(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 4, 4)
	const attemptStatement = "max(member.item_attempt)"
	trace := newReconcileQueryTrace(attemptStatement, runStatement)
	controller := harness.tracedController(t, ctx, "traced-dispatcher", trace)
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("activate round = (%t, %v)", worked, err)
	}
	trace.take()
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("dispatch batch = (%t, %v)", worked, err)
	}
	_, watched := trace.take()
	if watched[attemptStatement] != 1 {
		t.Fatalf("dispatching a batch of four read attempts in %d statements, want 1", watched[attemptStatement])
	}
	if watched[runStatement] != 0 {
		t.Fatalf("dispatch read %d Runs one by one", watched[runStatement])
	}
	executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("executions = (%+v, %v), want one submitted batch", executions, err)
	}
	members, err := harness.audits.ListExecutionItems(ctx, executions[0].ExecutionID)
	if err != nil || len(members) != 4 {
		t.Fatalf("batch members = (%d, %v), want 4", len(members), err)
	}
	for index, member := range members {
		if member.ItemAttempt != 1 || member.BatchOrdinal != index {
			t.Fatalf("member %d = attempt %d ordinal %d, want first attempts in order", index, member.ItemAttempt, member.BatchOrdinal)
		}
	}
}
