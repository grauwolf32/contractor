package auditcontroller

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// The fixed drain bound is a constant below the snapshot window its
// candidates come from (LIM-AUX-14).
func TestDrainBoundStaysWithinTheSnapshotWindow(t *testing.T) {
	if maxDrainMutationsPerStep < 1 || maxDrainMutationsPerStep > auditstore.MaxReconcileRows {
		t.Fatalf("drain bound %d is outside 1..%d", maxDrainMutationsPerStep, auditstore.MaxReconcileRows)
	}
	harness := newControllerHarness(t, 1, 1)
	if harness.controller.drainLimit != maxDrainMutationsPerStep {
		t.Fatalf("Controller drain bound = %d, want %d", harness.controller.drainLimit, maxDrainMutationsPerStep)
	}
}

// submitFakeChildren activates the Round and submits one running child Run
// per item.
func (h *controllerHarness) submitFakeChildren(t *testing.T, children int) []string {
	t.Helper()
	for pass := 0; pass <= children; pass++ {
		if worked, err := h.controller.RunOnce(h.ctx); err != nil || !worked {
			t.Fatalf("submission pass %d = (%t, %v)", pass, worked, err)
		}
	}
	if got := h.creator.createdCount(); got != children {
		t.Fatalf("submitted %d children, want %d", got, children)
	}
	h.runs.mu.Lock()
	defer h.runs.mu.Unlock()
	return append([]string(nil), h.runs.order...)
}

func (s *fakeControllerStore) executionsIn(state auditstore.ExecutionState) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	count := 0
	for _, execution := range s.executions {
		if execution.State == state {
			count++
		}
	}
	return count
}

// Already-terminal children are observed in full batches and a tail, one
// batch per pass, and nothing else happens in a pass that observed.
func TestControllerObservesTerminalChildrenInBoundedBatches(t *testing.T) {
	children := 2*maxDrainMutationsPerStep + 6
	harness := newControllerHarness(t, children, children)
	for _, runID := range harness.submitFakeChildren(t, children) {
		harness.runs.finish(runID, runstore.RunSucceeded)
	}
	for pass, want := range []int{maxDrainMutationsPerStep, 2 * maxDrainMutationsPerStep, children} {
		if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
			t.Fatalf("observation pass %d = (%t, %v)", pass, worked, err)
		}
		if got := harness.store.executionsIn(auditstore.ExecutionCollecting); got != want {
			t.Fatalf("after observation pass %d, %d executions are collecting, want %d", pass, got, want)
		}
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || worked {
		t.Fatalf("pass after the last batch = (%t, %v), want no change", worked, err)
	}
}

// claimStealingStore lets another holder lease the Audit after the
// controller's batch committed a number of observations.
type claimStealingStore struct {
	*fakeControllerStore
	after    int
	observed int
}

func (s *claimStealingStore) ObserveTerminal(ctx context.Context, params auditstore.ObserveTerminalParams) (auditstore.Execution, error) {
	execution, err := s.fakeControllerStore.ObserveTerminal(ctx, params)
	if err == nil {
		s.observed++
		if s.observed == s.after {
			s.mu.Lock()
			s.epoch++
			s.mu.Unlock()
		}
	}
	return execution, err
}

// A claim lost in the middle of a batch ends it: the observations already
// committed stay, none is repeated, and the next holder continues.
func TestControllerDrainBatchStopsAtALostClaim(t *testing.T) {
	children := 12
	harness := newControllerHarness(t, children, children)
	for _, runID := range harness.submitFakeChildren(t, children) {
		harness.runs.finish(runID, runstore.RunFailed)
	}
	stealing := &claimStealingStore{fakeControllerStore: harness.store, after: 5}
	harness.controller.store = stealing
	worked, err := harness.controller.RunOnce(harness.ctx)
	if !worked || !errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("pass that lost its claim = (%t, %v), want progress and ErrClaimLost", worked, err)
	}
	if got := harness.store.executionsIn(auditstore.ExecutionCollecting); got != 5 {
		t.Fatalf("observations before the claim was lost = %d, want 5", got)
	}
	harness.store.mu.Lock()
	harness.store.held = false
	harness.store.mu.Unlock()
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("next holder pass = (%t, %v)", worked, err)
	}
	if got := harness.store.executionsIn(auditstore.ExecutionCollecting); got != children || stealing.observed != children {
		t.Fatalf("collecting = %d after %d observations, want each of %d observed once", got, stealing.observed, children)
	}
}

// racingSubmissionStore binds a Run to a later intent after the batch
// failed some intents, as a submission that won the race would.
type racingSubmissionStore struct {
	*fakeControllerStore
	runs     *fakeControllerRuns
	after    int
	failed   int
	racingID string
}

func (s *racingSubmissionStore) ObserveSubmissionFailure(
	ctx context.Context, params auditstore.ObserveSubmissionFailureParams,
) (auditstore.Execution, error) {
	execution, err := s.fakeControllerStore.ObserveSubmissionFailure(ctx, params)
	if err == nil {
		s.failed++
		if s.failed == s.after {
			s.runs.mu.Lock()
			s.runs.runs["run-racing"] = runstore.WorkflowRun{RunID: "run-racing", State: runstore.RunRunning}
			s.runs.order = append(s.runs.order, "run-racing")
			s.runs.mu.Unlock()
			if bindErr := s.bind(s.racingID, "run-racing"); bindErr != nil {
				return execution, bindErr
			}
		}
	}
	return execution, err
}

// createFakeIntents records one unbound execution intent per item, as a
// dispatch does before its Run exists.
func (h *controllerHarness) createFakeIntents(t *testing.T, intents int) []string {
	t.Helper()
	claims, err := h.store.Claim(h.ctx, auditstore.ClaimParams{HolderID: "intent-fixture", Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim = (%+v, %v)", claims, err)
	}
	h.store.mu.Lock()
	h.store.round.State = auditstore.RoundExecuting
	h.store.mu.Unlock()
	executionIDs := make([]string, 0, intents)
	for intent := range intents {
		snapshot, err := h.store.GetReconcileSnapshot(h.ctx, claims[0])
		if err != nil {
			t.Fatal(err)
		}
		prepared, err := (fakeSubmissionBuilder{}).Prepare(h.ctx, snapshot, snapshot.ReadyItems[0], 1)
		if err != nil {
			t.Fatal(err)
		}
		prepared.Intent.Claim = claims[0]
		execution, created, err := h.store.CreateExecutionIntent(h.ctx, prepared.Intent)
		if err != nil || !created {
			t.Fatalf("intent %d = (%t, %v)", intent, created, err)
		}
		executionIDs = append(executionIDs, execution.ExecutionID)
	}
	if err := h.store.ReleaseClaim(h.ctx, claims[0]); err != nil {
		t.Fatal(err)
	}
	return executionIDs
}

// A candidate that became stale in the middle of a batch ends it without
// repeating any effect: a Run bound to a later intent makes that intent's
// failure a stale precondition, and the bound Run joins the cancellation
// set instead.
func TestControllerDrainBatchStopsAtAStaleIntent(t *testing.T) {
	intents := 10
	harness := newControllerHarness(t, intents, intents)
	executionIDs := harness.createFakeIntents(t, intents)
	harness.store.cancelAudit()
	racing := &racingSubmissionStore{
		fakeControllerStore: harness.store, runs: harness.runs, after: 3, racingID: executionIDs[4],
	}
	harness.controller.store = racing
	worked, err := harness.controller.RunOnce(harness.ctx)
	if !worked || !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("pass with a stale intent = (%t, %v), want progress and ErrPrecondition", worked, err)
	}
	if racing.failed != 4 || harness.store.executionsIn(auditstore.ExecutionIntent) != 5 ||
		harness.store.executionsIn(auditstore.ExecutionSubmitted) != 1 {
		t.Fatalf("failed %d intents, %d intents and %d submitted remain; want 4 failed before the stale one",
			racing.failed, harness.store.executionsIn(auditstore.ExecutionIntent),
			harness.store.executionsIn(auditstore.ExecutionSubmitted))
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("remaining intents pass = (%t, %v)", worked, err)
	}
	if racing.failed != intents-1 || harness.store.executionsIn(auditstore.ExecutionIntent) != 0 {
		t.Fatalf("failed %d intents, want every unbound one", racing.failed)
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("cancel the racing Run = (%t, %v)", worked, err)
	}
	if got := harness.runs.cancelCount(); got != 1 {
		t.Fatalf("cancellation requests = %d, want the racing Run once", got)
	}
}

// budgetEndingRuns ends the step's budget while one cancellation request is
// in flight, which then fails as a request cut by its deadline does.
type budgetEndingRuns struct {
	*fakeControllerRuns
	end       context.CancelFunc
	at        int
	requested int
}

func (r *budgetEndingRuns) RequestRunCancellation(
	ctx context.Context, runID string, cancellation runstore.WorkflowRunCancellation,
) (runstore.WorkflowRun, error) {
	r.requested++
	if r.requested == r.at {
		r.end()
		return runstore.WorkflowRun{}, ctx.Err()
	}
	return r.fakeControllerRuns.RequestRunCancellation(ctx, runID, cancellation)
}

// When the step's budget ends under a batch, the batch stops: the requests
// it committed are progress, the interrupted one is left to the next step
// and no further request starts.
func TestControllerDrainBatchStopsWhenTheStepBudgetEnds(t *testing.T) {
	children := 10
	harness := newControllerHarness(t, children, children)
	harness.submitFakeChildren(t, children)
	harness.store.cancelAudit()
	ctx, end := context.WithCancel(harness.ctx)
	defer end()
	runs := &budgetEndingRuns{fakeControllerRuns: harness.runs, end: end, at: 4}
	harness.controller.runs = runs
	if worked, err := harness.controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("pass whose budget ended = (%t, %v), want progress without error", worked, err)
	}
	if runs.requested != 4 || harness.runs.cancelCount() != 3 {
		t.Fatalf("started %d requests and committed %d, want 4 started and 3 committed",
			runs.requested, harness.runs.cancelCount())
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("next pass = (%t, %v)", worked, err)
	}
	if got := harness.runs.cancelCount(); got != children {
		t.Fatalf("cancellation requests = %d, want every running child once", got)
	}
}

// A child Run that cannot be deleted yet ends the deletion batch; the
// deletions before it stay, a pass that deletes nothing reports no change,
// and the purge waits for a pass that finds no child Run left.
func TestControllerDeletionBatchStopsAtABlockedRun(t *testing.T) {
	children := 5
	harness := newControllerHarness(t, children, children)
	runIDs := harness.submitFakeChildren(t, children)
	for index, runID := range runIDs {
		if index != 2 {
			harness.runs.finish(runID, runstore.RunCancelled)
		}
	}
	harness.store.mu.Lock()
	harness.store.audit.State = auditstore.AuditDeleting
	harness.store.audit.Dispatch = auditstore.DispatchClosed
	harness.store.audit.OutstandingRunCount = 0
	harness.store.items, harness.store.executions = nil, nil
	harness.store.mu.Unlock()
	live := func() int {
		harness.runs.mu.Lock()
		defer harness.runs.mu.Unlock()
		return len(harness.runs.runs)
	}

	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked || live() != 3 {
		t.Fatalf("deletion pass = (%t, %v) with %d Runs left, want the two before the blocked Run deleted", worked, err, live())
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || worked || live() != 3 {
		t.Fatalf("blocked pass = (%t, %v) with %d Runs left, want no change", worked, err, live())
	}
	harness.runs.finish(runIDs[2], runstore.RunCancelled)
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked || live() != 0 {
		t.Fatalf("deletion pass = (%t, %v) with %d Runs left, want the rest deleted", worked, err, live())
	}
	if state := harness.store.auditSnapshot().State; state != auditstore.AuditDeleting {
		t.Fatalf("Audit purged in the pass that deleted its last Runs: %s", state)
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("purge pass = (%t, %v)", worked, err)
	}
	if state := harness.store.auditSnapshot().State; state != auditstore.AuditCompleted {
		t.Fatalf("purged fake Audit is %s", state)
	}
}

// steppingClock is a test clock that moves only when the test advances it.
type steppingClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *steppingClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *steppingClock) After(time.Duration) <-chan time.Time { return make(chan time.Time) }

func (c *steppingClock) advance(duration time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(duration)
}

// slowDeleter takes step of the clock for every child Run deletion.
type slowDeleter struct {
	RunDeleter
	clock *steppingClock
	step  time.Duration
}

func (d *slowDeleter) DeleteReleasedTerminalRun(ctx context.Context, ownerID, runID string) error {
	d.clock.advance(d.step)
	return d.RunDeleter.DeleteReleasedTerminalRun(ctx, ownerID, runID)
}

// A deleting step that outlasts the ordinary operation budget is logged with
// its Audit, so the delay it imposes on the Audits claimed with it is
// detected; a step within the budget logs nothing.
func TestControllerLogsADeletionStepBeyondTheOperationBudget(t *testing.T) {
	for _, test := range []struct {
		name string
		step time.Duration
		want bool
	}{
		{name: "within the budget", step: 10 * time.Millisecond, want: false},
		{name: "beyond the budget", step: 50 * time.Millisecond, want: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			children := 4
			harness := newControllerHarness(t, children, children)
			for _, runID := range harness.submitFakeChildren(t, children) {
				harness.runs.finish(runID, runstore.RunCancelled)
			}
			harness.store.mu.Lock()
			harness.store.audit.State = auditstore.AuditDeleting
			harness.store.audit.Dispatch = auditstore.DispatchClosed
			harness.store.audit.OutstandingRunCount = 0
			harness.store.items, harness.store.executions = nil, nil
			harness.store.mu.Unlock()
			clock := &steppingClock{now: time.Date(2026, 10, 24, 0, 0, 0, 0, time.UTC)}
			var logs bytes.Buffer
			harness.controller.options.Clock = clock
			harness.controller.options.Logger = slog.New(slog.NewTextHandler(&logs, nil))
			harness.controller.deleter = &slowDeleter{RunDeleter: harness.runs, clock: clock, step: test.step}

			if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
				t.Fatalf("deletion pass = (%t, %v)", worked, err)
			}
			logged := strings.Contains(logs.String(), "Audit deletion step outlasted the ordinary operation budget")
			if logged != test.want || (logged && !strings.Contains(logs.String(), "audit_id="+harness.store.audit.AuditID)) {
				t.Fatalf("deletion of %d Runs taking %s each logged %q, want a warning: %t",
					children, test.step, logs.String(), test.want)
			}
		})
	}
}
