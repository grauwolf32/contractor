//go:build integration

package auditcontroller

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Statement fragments that identify, in a traced Controller pass, its
// reconcile snapshot, one drain mutation of each batched phase, one
// collection receipt or one Audit purge batch.
const (
	snapshotStatement          = "LEFT JOIN workflow_runs AS run ON run.run_id = execution.run_id"
	cancellationStatement      = "state_reason_code = 'user_cancelled'"
	observationStatement       = "terminal_outcome = observed.terminal_outcome"
	submissionFailureStatement = "terminal_outcome = 'submission-failed'"
	collectionStatement        = "collection_receipt_id = $5, collection_disposition = $6"
	runDeletionStatement       = "DELETE FROM workflow_runs\nWHERE run_id = $1 AND owner_id = $2"
	purgeStatement             = "set_config('contractor.lifecycle_purge', 'audit', true)"
)

// batchedDrainStatements are the mutations a reconcile step batches under
// the fixed per-step bound (AUX-R96).
var batchedDrainStatements = []string{
	cancellationStatement, observationStatement, submissionFailureStatement, runDeletionStatement,
}

// drainPass is what one traced Controller pass did.
type drainPass struct {
	worked     bool
	statements int
	counts     map[string]int
}

// kind names the one batched mutation, collection or purge the pass made,
// or "other" for a pass that changed something else.
func (p drainPass) kind(t *testing.T) string {
	t.Helper()
	found := ""
	for _, fragment := range append([]string{collectionStatement, purgeStatement}, batchedDrainStatements...) {
		if p.counts[fragment] == 0 {
			continue
		}
		if found != "" {
			t.Fatalf("one pass made both %q and %q", found, fragment)
		}
		found = fragment
	}
	if found == "" {
		return "other"
	}
	return found
}

// drainRecorder drives a Controller whose every statement it traces, one
// pass at a time.
type drainRecorder struct {
	t          *testing.T
	ctx        context.Context
	trace      *reconcileQueryTrace
	controller *Controller
	pool       *pgxpool.Pool
}

// drainRecorder returns a recorder of a collecting Controller of holderID
// whose drain batches are bounded by limit.
func (h *postgresControllerHarness) drainRecorder(
	t *testing.T, ctx context.Context, holderID string, limit int,
) *drainRecorder {
	t.Helper()
	trace := newReconcileQueryTrace(append([]string{
		snapshotStatement, collectionStatement, purgeStatement,
	}, batchedDrainStatements...)...)
	configuration := h.pool.Config()
	configuration.ConnConfig.Tracer = trace
	pool, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	controller := h.collectingController(t, pool, holderID, 1)
	controller.drainLimit = limit
	return &drainRecorder{t: t, ctx: ctx, trace: trace, controller: controller, pool: pool}
}

// collectingController reconciles through stores on pool, collecting with
// the ordinary importer and claiming up to claimBatch Audits per pass. Run
// creation and submission preparation keep the harness pool.
func (h *postgresControllerHarness) collectingController(
	t *testing.T, pool *pgxpool.Pool, holderID string, claimBatch int,
) *Controller {
	t.Helper()
	audits := auditstoretest.NewStore(pool).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	runs := runstore.NewPostgresStore(pool)
	access, err := auditimport.NewArtifactAccess(
		artifacts.NewService(artifacts.NewPostgresRepository(pool)), hookstore.NewClaimedArtifacts(pool),
	)
	if err != nil {
		t.Fatal(err)
	}
	collector, err := auditimport.New(audits, audits, runs, access)
	if err != nil {
		t.Fatal(err)
	}
	var ids int
	controller, err := New(
		audits, audits, runs, rundeletion.New(pool), h.runService, h.builder(t), &postgresNotifier{},
		Options{
			HolderID: holderID, ClaimLease: 20 * time.Second,
			OperationTimeout: 10 * time.Second, ClaimBatch: claimBatch, Collector: collector,
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

func (r *drainRecorder) pass() drainPass {
	r.t.Helper()
	r.trace.take()
	worked, err := r.controller.RunOnce(r.ctx)
	if err != nil {
		r.t.Fatalf("Controller pass: %v", err)
	}
	total, counts := r.trace.take()
	return drainPass{worked: worked, statements: total, counts: counts}
}

// untilIdle runs passes until one changes nothing and returns the passes
// that changed something.
func (r *drainRecorder) untilIdle() []drainPass {
	r.t.Helper()
	passes := make([]drainPass, 0)
	for range 2000 {
		pass := r.pass()
		if !pass.worked {
			return passes
		}
		passes = append(passes, pass)
	}
	r.t.Fatal("the Controller did not become idle")
	return nil
}

// until runs passes until done reports true after one, and returns them.
func (r *drainRecorder) until(done func() bool) []drainPass {
	r.t.Helper()
	passes := make([]drainPass, 0)
	for range 2000 {
		pass := r.pass()
		passes = append(passes, pass)
		if done() {
			return passes
		}
		if !pass.worked {
			r.t.Fatal("the Controller became idle before reaching the expected state")
		}
	}
	r.t.Fatal("the Controller did not reach the expected state")
	return nil
}

// drainProfile summarizes passes: the mutation count of every pass of each
// batched phase, the passes of every other kind, the reconcile snapshots
// read and, per batched phase, the most statements one of its passes issued.
type drainProfile struct {
	batches       map[string][]int
	passes        map[string]int
	snapshots     int
	maxStatements map[string]int
}

func profileDrain(t *testing.T, passes []drainPass) drainProfile {
	t.Helper()
	profile := drainProfile{
		batches: map[string][]int{}, passes: map[string]int{}, maxStatements: map[string]int{},
	}
	for _, pass := range passes {
		profile.snapshots += pass.counts[snapshotStatement]
		kind := pass.kind(t)
		profile.passes[kind]++
		switch kind {
		case collectionStatement:
			if pass.counts[kind] != 1 {
				t.Fatalf("one pass collected %d executions, want one", pass.counts[kind])
			}
		case purgeStatement, "other":
		default:
			profile.batches[kind] = append(profile.batches[kind], pass.counts[kind])
			profile.maxStatements[kind] = max(profile.maxStatements[kind], pass.statements)
		}
	}
	return profile
}

// batchSizes is the mutation count of each pass that drains n eligible
// children under a bound of limit: full batches and then the tail.
func batchSizes(n, limit int) []int {
	sizes := make([]int, 0)
	for n > 0 {
		size := min(n, limit)
		sizes = append(sizes, size)
		n -= size
	}
	return sizes
}

func requireBatches(t *testing.T, profile drainProfile, fragment string, want []int) {
	t.Helper()
	got := profile.batches[fragment]
	if len(got) != len(want) {
		t.Fatalf("%q passes = %v, want %v", fragment, got, want)
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("%q passes = %v, want %v", fragment, got, want)
		}
	}
}

// clearOutstanding lets the next dispatch of a controlled fixture reserve
// beyond the dispatch window. The window (LIM-AUX-1) never holds more than
// 32 outstanding children per lane class, so a fixture with more submitted
// or intended children clears the counter before each dispatch and
// restoreOutstanding sets it to the true count afterwards; every intent, Run
// and binding still comes from the ordinary paths.
func (h *postgresControllerHarness) clearOutstanding(t *testing.T, ctx context.Context) {
	t.Helper()
	if _, err := h.pool.Exec(ctx, `
UPDATE audits SET outstanding_run_count = 0, outstanding_tool_run_count = 0
 WHERE audit_id = $1`, h.started.Audit.AuditID); err != nil {
		t.Fatal(err)
	}
}

func (h *postgresControllerHarness) restoreOutstanding(t *testing.T, ctx context.Context) {
	t.Helper()
	if _, err := h.pool.Exec(ctx, `
UPDATE audits SET outstanding_run_count = (
       SELECT count(*) FROM audit_executions
        WHERE audit_id = $1 AND state IN ('intent', 'submitted'))
 WHERE audit_id = $1`, h.started.Audit.AuditID); err != nil {
		t.Fatal(err)
	}
}

// submitChildren activates the Round and submits one child Run per item
// through the ordinary Controller dispatch, and returns the Runs in
// submission order.
func (h *postgresControllerHarness) submitChildren(t *testing.T, ctx context.Context, children int) []string {
	t.Helper()
	dispatcher := h.controller(t)
	dispatcher.options.OperationTimeout = 10 * time.Second
	dispatcher.options.ClaimLease = 20 * time.Second
	if worked, err := dispatcher.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("activate Round = (%t, %v)", worked, err)
	}
	for child := range children {
		h.clearOutstanding(t, ctx)
		if worked, err := dispatcher.RunOnce(ctx); err != nil || !worked {
			t.Fatalf("submit child %d = (%t, %v)", child, worked, err)
		}
	}
	h.restoreOutstanding(t, ctx)
	executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(executions) != children {
		t.Fatalf("submitted executions = (%d, %v), want %d", len(executions), err, children)
	}
	runIDs := make([]string, children)
	for index, execution := range executions {
		if execution.State != auditstore.ExecutionSubmitted || execution.RunID == nil {
			t.Fatalf("execution %s is %s, want a submitted child", execution.ExecutionID, execution.State)
		}
		runIDs[index] = *execution.RunID
	}
	return runIDs
}

// createUnboundIntents activates the Round and records one execution intent
// per item through the claimed store, as a dispatch does before its Run
// exists.
func (h *postgresControllerHarness) createUnboundIntents(t *testing.T, ctx context.Context, intents int) {
	t.Helper()
	if worked, err := h.controller(t).RunOnce(ctx); err != nil || !worked {
		t.Fatalf("activate Round = (%t, %v)", worked, err)
	}
	claims, err := h.audits.Claim(ctx, auditstore.ClaimParams{
		HolderID: "intent-fixture", Lease: time.Minute, Limit: 1,
	})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	builder := h.builder(t).WithClaim(claim)
	for intent := range intents {
		h.clearOutstanding(t, ctx)
		snapshot, err := h.audits.GetReconcileSnapshot(ctx, claim)
		if err != nil || len(snapshot.ReadyItems) == 0 {
			t.Fatalf("intent %d snapshot = (%d ready, %v)", intent, len(snapshot.ReadyItems), err)
		}
		prepared, err := builder.PrepareBatch(ctx, snapshot, []CheckExecutionMember{
			{Item: snapshot.ReadyItems[0], Attempt: 1},
		})
		if err != nil {
			t.Fatal(err)
		}
		prepared.Intent.Claim = claim
		if _, created, err := h.audits.CreateExecutionIntent(ctx, prepared.Intent); err != nil || !created {
			t.Fatalf("intent %d = (%t, %v)", intent, created, err)
		}
	}
	h.restoreOutstanding(t, ctx)
	if err := h.audits.ReleaseClaim(ctx, claim); err != nil {
		t.Fatal(err)
	}
}

// ownerMutation applies the owner's cancel or delete to the Audit's current
// revision.
func (h *postgresControllerHarness) ownerMutation(t *testing.T, ctx context.Context, operation string) auditstore.Audit {
	t.Helper()
	current, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	params := auditservice.MutationParams{
		OwnerID: current.OwnerID, AuditID: current.AuditID, ExpectedRevision: current.Revision,
		IdempotencyKey: operation + "-audit", RequestDigest: postgresDigest(operation + "-audit"),
	}
	var result auditservice.MutationResult
	switch operation {
	case "cancel":
		result, err = h.service.Cancel(ctx, params)
	case "delete":
		result, err = h.service.Delete(ctx, params)
	default:
		t.Fatalf("unknown owner mutation %q", operation)
	}
	if err != nil || result.Audit.State != auditstore.AuditCancelling {
		t.Fatalf("owner %s = (%+v, %v)", operation, result.Audit, err)
	}
	return result.Audit
}

func (h *postgresControllerHarness) finishRuns(
	t *testing.T, ctx context.Context, runIDs []string, from, to runstore.WorkflowRunState,
) {
	t.Helper()
	for _, runID := range runIDs {
		if _, err := h.runs.TransitionRun(ctx, runID, from, to, runstore.Reason{Code: "test_finished"}); err != nil {
			t.Fatalf("finish child Run %s: %v", runID, err)
		}
	}
}

func (h *postgresControllerHarness) auditState(t *testing.T, ctx context.Context) auditstore.AuditState {
	t.Helper()
	audit, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if errors.Is(err, auditstore.ErrNotFound) {
		return ""
	}
	if err != nil {
		t.Fatal(err)
	}
	return audit.State
}

// requireReceipts checks that collected executions of the Audit hold their
// receipt and that no execution was observed or collected twice.
func (h *postgresControllerHarness) requireReceipts(t *testing.T, ctx context.Context, collected int) {
	t.Helper()
	var receipts, events, repeated int
	if err := h.pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM audit_executions
         WHERE audit_id = $1 AND state = 'collected' AND collection_receipt_id IS NOT NULL),
       (SELECT count(*) FROM audit_events WHERE audit_id = $1 AND kind = 'execution.collected'),
       (SELECT count(*) FROM (
            SELECT 1 FROM audit_events
             WHERE audit_id = $1 AND kind IN ('execution.terminal_observed', 'execution.collected')
             GROUP BY kind, entity_id HAVING count(*) > 1) AS repeated)`,
		h.started.Audit.AuditID).Scan(&receipts, &events, &repeated); err != nil {
		t.Fatal(err)
	}
	if receipts != collected || events != collected || repeated != 0 {
		t.Fatalf("receipts = %d, collection events = %d, repeated observations or collections = %d; want %d once each",
			receipts, events, repeated, collected)
	}
}

// fullBatchStatements records, per batched phase, the statements of the
// largest pass of each size of Audit, to show that a pass's cost does not
// grow with the Audit beyond its fixed bound.
type fullBatchStatements struct {
	mu     sync.Mutex
	byKind map[string]map[int]int
}

func (s *fullBatchStatements) record(children int, profile drainProfile) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for kind, statements := range profile.maxStatements {
		if s.byKind[kind] == nil {
			s.byKind[kind] = map[int]int{}
		}
		s.byKind[kind][children] = max(s.byKind[kind][children], statements)
	}
}

// drainCase is one Audit size and the drain bound a test applies to it; a
// bound of one reproduces the earlier one-mutation-per-pass baseline.
type drainCase struct {
	children int
	limit    int
}

func (c drainCase) name() string {
	if c.limit == 1 {
		return fmt.Sprintf("%d children one per pass", c.children)
	}
	return fmt.Sprintf("%d children", c.children)
}

var drainCases = []drainCase{
	{children: 1, limit: maxDrainMutationsPerStep},
	{children: 50, limit: maxDrainMutationsPerStep},
	{children: 200, limit: maxDrainMutationsPerStep},
	{children: 50, limit: 1},
}

// requireImprovement compares the snapshots of the bounded drain of each
// size with its one-per-pass baseline: equal for one child, and fewer by
// exactly the passes the batches saved otherwise.
func requireImprovement(t *testing.T, snapshots map[drainCase]int, saved map[drainCase]int) {
	t.Helper()
	for _, candidate := range drainCases {
		if candidate.limit == 1 {
			continue
		}
		baseline, measured := snapshots[drainCase{children: candidate.children, limit: 1}]
		batched := snapshots[candidate]
		if !measured {
			continue
		}
		if baseline-batched != saved[candidate] {
			t.Fatalf("%d children read %d snapshots one per pass and %d in batches, want %d saved",
				candidate.children, baseline, batched, saved[candidate])
		}
	}
	for _, candidate := range drainCases {
		if candidate.limit != 1 && candidate.children > 1 && saved[candidate] <= 0 {
			t.Fatalf("%d children saved %d snapshots, want fewer than one per mutation",
				candidate.children, saved[candidate])
		}
	}
}

// savedPasses is how many fewer passes the batches of sizes need than one
// pass per mutation.
func savedPasses(phases ...[]int) int {
	saved := 0
	for _, sizes := range phases {
		for _, size := range sizes {
			saved += size - 1
		}
	}
	return saved
}

// TestPostgresControllerDrainBatchesCancellationObservationAndDeletion
// deletes an active Audit with 1, 50 and 200 running children. Each
// claimed pass requests at most the fixed bound of cancellations, records at
// most that many terminal observations and deletes at most that many
// released Runs, while collection, transitions and purge batches keep their
// own passes. Nonterminal children keep the settlement barrier closed until
// the test transitions them, and every child is observed and collected
// exactly once before the Audit leaves cancelling.
func TestPostgresControllerDrainBatchesCancellationObservationAndDeletion(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
	defer cancel()
	statements := &fullBatchStatements{byKind: map[string]map[int]int{}}
	snapshots, saved := map[drainCase]int{}, map[drainCase]int{}
	for _, drain := range drainCases {
		t.Run(drain.name(), func(t *testing.T) {
			children := drain.children
			harness := newPostgresControllerReviewHarness(t, ctx, 0, children)
			runIDs := harness.submitChildren(t, ctx, children)
			harness.ownerMutation(t, ctx, "delete")
			recorder := harness.drainRecorder(t, ctx, "drain-controller", drain.limit)

			// Cancellation requests come in bounded batches; the dispatch
			// hold is released after them. Requests are not terminal
			// transitions: the barrier stays closed.
			requests := profileDrain(t, recorder.untilIdle())
			requireBatches(t, requests, cancellationStatement, batchSizes(children, drain.limit))
			if requests.passes["other"] != 1 || len(requests.passes) != 2 {
				t.Fatalf("cancellation passes = %v, want the requests and one hold release", requests.passes)
			}
			if state := harness.auditState(t, ctx); state != auditstore.AuditCancelling {
				t.Fatalf("Audit with cancelling children is %q, want cancelling", state)
			}
			for _, runID := range runIDs {
				run, err := harness.runs.GetRun(ctx, runID)
				if err != nil || run.State != runstore.RunCancelling {
					t.Fatalf("child Run %s = (%s, %v), want cancelling", runID, run.State, err)
				}
			}

			// All but the last child finish: their observations come in
			// bounded batches and each is collected in its own pass, but the
			// last running child keeps the Audit cancelling.
			early := runIDs[:children-1]
			harness.finishRuns(t, ctx, early, runstore.RunCancelling, runstore.RunCancelled)
			finished := profileDrain(t, recorder.untilIdle())
			requireBatches(t, finished, observationStatement, batchSizes(len(early), drain.limit))
			if finished.passes[collectionStatement] != len(early) ||
				len(finished.passes) != min(2, 2*len(early)) {
				t.Fatalf("early completion passes = %v, want %d collections", finished.passes, len(early))
			}
			if state := harness.auditState(t, ctx); state != auditstore.AuditCancelling {
				t.Fatalf("Audit with a running child is %q, want cancelling", state)
			}

			// The last child finishes and the Audit settles to deleting once
			// every child holds exactly one receipt.
			harness.requireReceipts(t, ctx, len(early))
			harness.finishRuns(t, ctx, runIDs[children-1:], runstore.RunCancelling, runstore.RunCancelled)
			settled := profileDrain(t, recorder.until(func() bool {
				return harness.auditState(t, ctx) == auditstore.AuditDeleting
			}))
			requireBatches(t, settled, observationStatement, []int{1})
			if settled.passes[collectionStatement] != 1 || settled.passes["other"] != 1 || len(settled.passes) != 3 {
				t.Fatalf("settlement passes = %v, want one observation, collection and transition", settled.passes)
			}
			harness.requireReceipts(t, ctx, children)

			// Released Runs are deleted in bounded batches; the purge follows
			// in its own passes.
			deleted := profileDrain(t, recorder.untilIdle())
			requireBatches(t, deleted, runDeletionStatement, batchSizes(children, drain.limit))
			if deleted.passes[purgeStatement] < 1 || len(deleted.passes) != 2 {
				t.Fatalf("deletion passes = %v, want the deletions and the purge", deleted.passes)
			}
			if state := harness.auditState(t, ctx); state != "" {
				t.Fatalf("deleted Audit is %q", state)
			}
			for _, runID := range runIDs {
				if _, err := harness.runs.GetRun(ctx, runID); !errors.Is(err, runstore.ErrNotFound) {
					t.Fatalf("deleted child Run %s lookup = %v", runID, err)
				}
			}

			total := requests.snapshots + finished.snapshots + settled.snapshots + deleted.snapshots
			wantTotal := len(batchSizes(children, drain.limit)) + 1 + // requests, hold release
				len(batchSizes(len(early), drain.limit)) + len(early) + // early completions
				1 + 1 + 1 + // last observation, collection and the transition to deleting
				len(batchSizes(children, drain.limit)) + deleted.passes[purgeStatement]
			if total != wantTotal {
				t.Fatalf("drain read %d snapshots, want %d", total, wantTotal)
			}
			snapshots[drain] = total
			saved[drain] = savedPasses(
				batchSizes(children, drain.limit), batchSizes(len(early), drain.limit),
				batchSizes(children, drain.limit),
			)
			if drain.limit != 1 {
				for _, profile := range []drainProfile{requests, finished, settled, deleted} {
					statements.record(children, profile)
				}
			}
			t.Logf("%s: %d snapshots; cancellations %v, observations %v and %v, deletions %v, %d purge passes; "+
				"most statements per batched pass %v %v %v",
				drain.name(), total, requests.batches[cancellationStatement],
				finished.batches[observationStatement], settled.batches[observationStatement],
				deleted.batches[runDeletionStatement], deleted.passes[purgeStatement],
				requests.maxStatements, finished.maxStatements, deleted.maxStatements)
		})
	}
	requireImprovement(t, snapshots, saved)
	statements.requireConstant(t)
}

// requireConstant checks that full batches of 50- and 200-child Audits cost
// the same statements: a pass's work is bounded by its batch, not the Audit.
func (s *fullBatchStatements) requireConstant(t *testing.T) {
	t.Helper()
	for kind, bySize := range s.byKind {
		if bySize[50] != bySize[200] {
			t.Fatalf("the largest %q pass issued %d statements with 50 children and %d with 200",
				kind, bySize[50], bySize[200])
		}
	}
}

// TestPostgresControllerDrainBatchesUnboundIntentFailures cancels an Audit
// whose 1, 50 or 200 execution intents never got a child Run. Each claimed
// pass fails at most the fixed bound of them; each failed intent is then
// collected in its own pass, and the Audit settles as cancelled with one
// receipt per execution.
func TestPostgresControllerDrainBatchesUnboundIntentFailures(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
	defer cancel()
	statements := &fullBatchStatements{byKind: map[string]map[int]int{}}
	snapshots, saved := map[drainCase]int{}, map[drainCase]int{}
	for _, drain := range drainCases {
		t.Run(drain.name(), func(t *testing.T) {
			children := drain.children
			harness := newPostgresControllerReviewHarness(t, ctx, 0, children)
			harness.createUnboundIntents(t, ctx, children)
			harness.ownerMutation(t, ctx, "cancel")
			recorder := harness.drainRecorder(t, ctx, "intent-drain-controller", drain.limit)

			profile := profileDrain(t, recorder.untilIdle())
			failures := batchSizes(children, drain.limit)
			requireBatches(t, profile, submissionFailureStatement, failures)
			if profile.passes[collectionStatement] != children {
				t.Fatalf("collection passes = %d, want one per execution", profile.passes[collectionStatement])
			}
			if state := harness.auditState(t, ctx); state != auditstore.AuditCancelled {
				t.Fatalf("drained Audit is %q, want cancelled", state)
			}
			harness.requireReceipts(t, ctx, children)
			wantTotal := len(failures) + children + profile.passes["other"]
			if profile.snapshots != wantTotal || profile.passes["other"] != intentSettlementPasses {
				t.Fatalf("drain read %d snapshots with %d other passes, want %d with %d",
					profile.snapshots, profile.passes["other"], len(failures)+children+intentSettlementPasses,
					intentSettlementPasses)
			}
			snapshots[drain] = profile.snapshots
			saved[drain] = savedPasses(failures)
			if drain.limit != 1 {
				statements.record(children, profile)
			}
			t.Logf("%s: %d snapshots; failures %v, %d collections, %d other passes; most statements per batched pass %v",
				drain.name(), profile.snapshots, failures, profile.passes[collectionStatement],
				profile.passes["other"], profile.maxStatements)
		})
	}
	requireImprovement(t, snapshots, saved)
	statements.requireConstant(t)
}

// intentSettlementPasses are the passes after the last collection of a
// cancelled Audit's failed intents: one settlement of the items their
// receipts returned to ready (at most 200 per pass), the dispatch hold
// release and the transition to cancelled.
const intentSettlementPasses = 3

// observationHookStore runs hook once, after the batch committed a number of
// terminal observations, while the Controller still reconciles.
type observationHookStore struct {
	Store
	after    int
	observed int
	hook     func() error
	hookErr  error
}

func (s *observationHookStore) ObserveTerminal(
	ctx context.Context, params auditstore.ObserveTerminalParams,
) (auditstore.Execution, error) {
	execution, err := s.Store.ObserveTerminal(ctx, params)
	if err == nil {
		s.observed++
		if s.observed == s.after {
			s.hookErr = s.hook()
		}
	}
	return execution, err
}

// executionsIn counts the Audit's executions in state.
func (h *postgresControllerHarness) executionsIn(
	t *testing.T, ctx context.Context, state auditstore.ExecutionState,
) int {
	t.Helper()
	var count int
	if err := h.pool.QueryRow(ctx, `
SELECT count(*) FROM audit_executions WHERE audit_id = $1 AND state = $2`,
		h.started.Audit.AuditID, string(state)).Scan(&count); err != nil {
		t.Fatal(err)
	}
	return count
}

// TestPostgresControllerDrainBatchStopsAtALostClaim lets another holder
// take the Audit's claim after a batch committed five of twelve terminal
// observations. The store refuses the sixth, the batch ends there, and the
// next holder observes the rest: every child is observed exactly once and
// the outstanding count reaches zero.
func TestPostgresControllerDrainBatchStopsAtALostClaim(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	children := 12
	harness := newPostgresControllerReviewHarness(t, ctx, 0, children)
	runIDs := harness.submitChildren(t, ctx, children)
	harness.finishRuns(t, ctx, runIDs, runstore.RunPending, runstore.RunFailed)

	first := harness.collectingController(t, harness.pool, "first-holder", 1)
	hooked := &observationHookStore{Store: first.store, after: 5, hook: func() error {
		if _, err := harness.pool.Exec(ctx, `
UPDATE audit_controller_claims SET expires_at = clock_timestamp() WHERE audit_id = $1`,
			harness.started.Audit.AuditID); err != nil {
			return err
		}
		claims, err := harness.audits.Claim(ctx, auditstore.ClaimParams{
			HolderID: "rival-holder", Lease: 20 * time.Second, Limit: 1,
		})
		if err != nil || len(claims) != 1 {
			return fmt.Errorf("rival claim = (%+v, %w)", claims, err)
		}
		return harness.audits.ReleaseClaim(ctx, claims[0])
	}}
	first.store = hooked
	worked, err := first.RunOnce(ctx)
	if hooked.hookErr != nil {
		t.Fatal(hooked.hookErr)
	}
	if !worked || !errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("pass that lost its claim = (%t, %v), want progress and ErrClaimLost", worked, err)
	}
	if got := harness.executionsIn(t, ctx, auditstore.ExecutionCollecting); got != 5 || hooked.observed != 5 {
		t.Fatalf("observed %d executions (%d collecting) before the claim was lost, want 5", hooked.observed, got)
	}

	next := harness.collectingController(t, harness.pool, "next-holder", 1)
	if worked, err := next.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("next holder pass = (%t, %v)", worked, err)
	}
	if got := harness.executionsIn(t, ctx, auditstore.ExecutionCollecting); got != children {
		t.Fatalf("collecting executions = %d, want %d", got, children)
	}
	harness.requireReceipts(t, ctx, 0)
	var observed, outstanding int
	if err := harness.pool.QueryRow(ctx, `
SELECT (SELECT count(DISTINCT entity_id) FROM audit_events
         WHERE audit_id = $1 AND kind = 'execution.terminal_observed'),
       (SELECT outstanding_run_count FROM audits WHERE audit_id = $1)`,
		harness.started.Audit.AuditID).Scan(&observed, &outstanding); err != nil {
		t.Fatal(err)
	}
	if observed != children || outstanding != 0 {
		t.Fatalf("observed executions = %d, outstanding Runs = %d; want %d and 0", observed, outstanding, children)
	}
}

// cancellationHookRuns counts the cancellation requests of each child Run
// and runs hook once, after a number of them, while the Controller still
// reconciles.
type cancellationHookRuns struct {
	RunStore
	after    int
	requests map[string]int
	total    int
	hook     func() error
	hookErr  error
}

func (r *cancellationHookRuns) RequestRunCancellation(
	ctx context.Context, runID string, cancellation runstore.WorkflowRunCancellation,
) (runstore.WorkflowRun, error) {
	run, err := r.RunStore.RequestRunCancellation(ctx, runID, cancellation)
	if err == nil {
		r.requests[runID]++
		r.total++
		if r.total == r.after {
			r.hookErr = r.hook()
		}
	}
	return run, err
}

// TestPostgresControllerDrainBatchKeepsRevisionFencedWorkForAFreshSnapshot
// lets the owner delete a cancelling Audit while a batch requests the
// cancellation of its twelve children. The requests stay valid under the
// new revision, so the batch requests each child once; the pass then ends,
// and only later passes, on fresh snapshots, release the dispatch hold and
// settle the Audit, which ends deleted as the owner's change requires.
func TestPostgresControllerDrainBatchKeepsRevisionFencedWorkForAFreshSnapshot(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	children := 12
	harness := newPostgresControllerReviewHarness(t, ctx, 0, children)
	runIDs := harness.submitChildren(t, ctx, children)
	cancelled := harness.ownerMutation(t, ctx, "cancel")

	controller := harness.collectingController(t, harness.pool, "revision-controller", 1)
	var deleted auditstore.Audit
	hooked := &cancellationHookRuns{RunStore: controller.runs, after: 3, requests: map[string]int{}, hook: func() error {
		result, err := harness.service.Delete(ctx, auditservice.MutationParams{
			OwnerID: cancelled.OwnerID, AuditID: cancelled.AuditID, ExpectedRevision: cancelled.Revision,
			IdempotencyKey: "delete-during-batch", RequestDigest: postgresDigest("delete-during-batch"),
		})
		deleted = result.Audit
		return err
	}}
	controller.runs = hooked
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("cancellation batch = (%t, %v)", worked, err)
	}
	if hooked.hookErr != nil {
		t.Fatal(hooked.hookErr)
	}
	if deleted.Revision <= cancelled.Revision || deleted.DeletionRequestedAt == nil {
		t.Fatalf("owner deletion during the batch = %+v, want a new revision with deletion requested", deleted)
	}
	if hooked.total != children || len(hooked.requests) != children {
		t.Fatalf("batch made %d requests for %d children, want each of %d once", hooked.total, len(hooked.requests), children)
	}
	after, err := harness.audits.Get(ctx, cancelled.OwnerID, cancelled.AuditID)
	if err != nil || after.Revision != deleted.Revision || after.Hold != cancelled.Hold {
		t.Fatalf("Audit after the batch = (%+v, %v), want the owner's revision and no further change", after, err)
	}

	harness.finishRuns(t, ctx, runIDs, runstore.RunCancelling, runstore.RunCancelled)
	for range 4 * children {
		if harness.auditState(t, ctx) == "" {
			break
		}
		if _, err := controller.RunOnce(ctx); err != nil {
			t.Fatal(err)
		}
	}
	if state := harness.auditState(t, ctx); state != "" {
		t.Fatalf("Audit deleted during the batch ended %q, want purged", state)
	}
	for runID, requests := range hooked.requests {
		if requests != 1 {
			t.Fatalf("child Run %s received %d cancellation requests, want 1", runID, requests)
		}
	}
}

// startSecondAudit starts another Audit with items checks in the harness
// Project.
func (h *postgresControllerHarness) startSecondAudit(
	t *testing.T, ctx context.Context, auditID string, items int,
) auditservice.StartedAudit {
	t.Helper()
	projectArtifacts, err := h.artifacts.Project(h.started.Audit.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	entries := make([]string, items)
	for index := range items {
		entries[index] = fmt.Sprintf(
			`{"key":"second-%d","version":"1","statement":"Verify second check %d.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}`,
			index, index,
		)
	}
	input, err := projectArtifacts.Write(
		ctx, contracts.ArtifactRef{Namespace: "inputs", Name: auditID + "-checklist"},
		artifacts.Payload{
			MediaType: "application/json",
			Data:      []byte(`{"schema":"contractor.audit.checklist.v1","items":[` + strings.Join(entries, ",") + `]}`),
		}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	draft, _, err := h.service.CreateDraft(ctx, auditservice.CreateDraftParams{
		Labels:  map[string]string{"cloud": "aws", "team": "appsec"},
		AuditID: auditID, OwnerID: h.started.Audit.OwnerID, ProjectID: h.started.Audit.ProjectID,
		Profile:       auditservice.ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:        map[string]contracts.ArtifactRef{"checklist": input.Ref},
		RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: "Verify the second checklist"},
		IdempotencyKey: "create-" + auditID, RequestDigest: postgresDigest("create-" + auditID),
	})
	if err != nil {
		t.Fatal(err)
	}
	started, err := h.service.Start(ctx, auditservice.StartParams{
		OwnerID: draft.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-" + auditID, RequestDigest: postgresDigest("start-" + auditID),
	})
	if err != nil {
		t.Fatal(err)
	}
	return started
}

// TestPostgresControllerSecondAuditProgressesBesideALargeDrain claims two
// Audits per pass: one with forty terminal children to observe and collect,
// one that still has to start its checks. Every pass gives each claimed
// Audit its turn, and the draining Audit's turn never exceeds the fixed
// bound of observations, so the other Audit activates its Round and submits
// its first child in the first two passes.
func TestPostgresControllerSecondAuditProgressesBesideALargeDrain(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	children := 40
	harness := newPostgresControllerReviewHarness(t, ctx, 0, children)
	runIDs := harness.submitChildren(t, ctx, children)
	harness.finishRuns(t, ctx, runIDs, runstore.RunPending, runstore.RunFailed)
	second := harness.startSecondAudit(t, ctx, "audit-controller-second", 2)
	controller := harness.collectingController(t, harness.pool, "two-audit-controller", 2)

	secondExecutions := func() int {
		executions, err := harness.audits.ListExecutions(ctx, second.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		return len(executions)
	}
	for pass, want := range []struct{ observed, secondExecutions int }{
		{maxDrainMutationsPerStep, 0}, {children, 1},
	} {
		if worked, err := controller.RunOnce(ctx); err != nil || !worked {
			t.Fatalf("pass %d = (%t, %v)", pass, worked, err)
		}
		if got := harness.executionsIn(t, ctx, auditstore.ExecutionCollecting); got != want.observed {
			t.Fatalf("after pass %d the draining Audit observed %d children, want %d", pass, got, want.observed)
		}
		if got := secondExecutions(); got != want.secondExecutions {
			t.Fatalf("after pass %d the second Audit has %d executions, want %d", pass, got, want.secondExecutions)
		}
	}
	round, err := harness.audits.GetRound(ctx, second.Audit.AuditID, second.Round.RoundID)
	if err != nil || round.State != auditstore.RoundExecuting {
		t.Fatalf("second Audit Round = (%s, %v), want executing", round.State, err)
	}
}
