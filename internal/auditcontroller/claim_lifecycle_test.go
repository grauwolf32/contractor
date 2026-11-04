package auditcontroller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"path/filepath"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// stepJoiningStore holds the reconcile step of its claim until the pass was
// cancelled and the test lets it end, and records whether the claim was
// released before or after the step ended (PER-R26).
type stepJoiningStore struct {
	*claimScript
	entered chan struct{}
	proceed chan struct{}

	mu             sync.Mutex
	events         []string
	releaseErr     error
	releaseBounded bool
}

func (s *stepJoiningStore) record(event string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, event)
}

func (s *stepJoiningStore) GetReconcileSnapshot(
	ctx context.Context, _ auditstore.ControllerClaim,
) (auditstore.ReconcileSnapshot, error) {
	close(s.entered)
	<-ctx.Done()
	<-s.proceed
	s.record("step ended")
	return auditstore.ReconcileSnapshot{}, ctx.Err()
}

func (s *stepJoiningStore) ReleaseClaim(ctx context.Context, claim auditstore.ControllerClaim) error {
	s.mu.Lock()
	s.events = append(s.events, "released")
	s.releaseErr = ctx.Err()
	_, s.releaseBounded = ctx.Deadline()
	s.mu.Unlock()
	return s.claimScript.ReleaseClaim(ctx, claim)
}

// A pass cancelled during a reconcile step joins the step before it releases
// the claim, and the release runs with a live context bounded by the
// operation budget, so shutdown gives the Audit back.
func TestControllerPassJoinsACancelledStepBeforeReleasingTheClaim(t *testing.T) {
	harness := newClaimHarness()
	// Only the cancellation may end the step, never its own budget.
	harness.controller.options.OperationTimeout = time.Minute
	store := &stepJoiningStore{
		claimScript: harness.store, entered: make(chan struct{}), proceed: make(chan struct{}),
	}
	harness.controller.store = store
	ctx, cancel := context.WithCancel(t.Context())
	type passResult struct {
		worked bool
		err    error
	}
	done := make(chan passResult, 1)
	go func() {
		worked, err := harness.controller.RunOnce(ctx)
		done <- passResult{worked: worked, err: err}
	}()
	<-store.entered
	cancel()
	close(store.proceed)
	result := <-done
	if result.worked || !errors.Is(result.err, context.Canceled) {
		t.Fatalf("cancelled pass = (%t, %v), want the step's cancellation", result.worked, result.err)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if !slices.Equal(store.events, []string{"step ended", "released"}) {
		t.Fatalf("events = %v, want the step joined before the release", store.events)
	}
	if store.releaseErr != nil || !store.releaseBounded {
		t.Fatalf("release context error = %v, bounded = %t, want a live bounded context",
			store.releaseErr, store.releaseBounded)
	}
	if harness.store.held {
		t.Fatal("the cancelled pass kept its claim")
	}
}

// concurrentClaimStore leases several Audits in one pass. Each step waits
// until every claimed Audit is in its step, then fails with its own error;
// each release reports the outcome scripted for its Audit.
type concurrentClaimStore struct {
	*claimScript
	audits     []string
	releaseErr map[string]error

	started    atomic.Int64
	allStarted chan struct{}
	mu         sync.Mutex
	released   []string
}

var errStepFailed = errors.New("reconcile step failed")

func (s *concurrentClaimStore) Claim(_ context.Context, params auditstore.ClaimParams) ([]auditstore.ControllerClaim, error) {
	claims := make([]auditstore.ControllerClaim, 0, len(s.audits))
	now := time.Date(2026, 10, 24, 0, 0, 0, 0, time.UTC)
	for _, auditID := range s.audits {
		claims = append(claims, auditstore.ControllerClaim{
			AuditID: auditID, HolderID: params.HolderID, Epoch: 1, ClaimedAt: now, ExpiresAt: now.Add(params.Lease),
		})
	}
	return claims, nil
}

func (s *concurrentClaimStore) GetReconcileSnapshot(
	ctx context.Context, claim auditstore.ControllerClaim,
) (auditstore.ReconcileSnapshot, error) {
	if s.started.Add(1) == int64(len(s.audits)) {
		close(s.allStarted)
	}
	select {
	case <-s.allStarted:
		return auditstore.ReconcileSnapshot{}, fmt.Errorf("%w: %s", errStepFailed, claim.AuditID)
	case <-ctx.Done():
		return auditstore.ReconcileSnapshot{}, ctx.Err()
	}
}

func (s *concurrentClaimStore) ReleaseClaim(_ context.Context, claim auditstore.ControllerClaim) error {
	s.mu.Lock()
	s.released = append(s.released, claim.AuditID)
	s.mu.Unlock()
	return s.releaseErr[claim.AuditID]
}

// Every Audit of a pass reconciles at the same time, so one Audit's step
// cannot hold back another's; each claim is released, a release that finds
// its claim lost is not a failure, and the remaining failures are reported
// per Audit in claim order.
func TestControllerPassReconcilesItsClaimsConcurrentlyAndReportsEachFailure(t *testing.T) {
	harness := newClaimHarness()
	// A sequential pass would wait out this budget instead of meeting the
	// other steps; the test fails on the error, it measures no time.
	harness.controller.options.OperationTimeout = 10 * time.Second
	errReleaseFailed := errors.New("release failed")
	store := &concurrentClaimStore{
		claimScript: harness.store,
		audits:      []string{"audit-a", "audit-b", "audit-c", "audit-d"},
		releaseErr: map[string]error{
			"audit-b": auditstore.ErrClaimLost,
			"audit-c": errReleaseFailed,
		},
		allStarted: make(chan struct{}),
	}
	harness.controller.store = store
	worked, err := harness.controller.RunOnce(t.Context())
	if worked || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("pass = (%t, %v), want concurrent steps and no lost-claim failure", worked, err)
	}
	store.mu.Lock()
	released := slices.Clone(store.released)
	store.mu.Unlock()
	slices.Sort(released)
	if !slices.Equal(released, store.audits) {
		t.Fatalf("released claims = %v, want %v", released, store.audits)
	}
	joined, ok := err.(interface{ Unwrap() []error })
	if !ok || len(joined.Unwrap()) != len(store.audits) {
		t.Fatalf("pass error = %v, want one failure per Audit", err)
	}
	for index, failure := range joined.Unwrap() {
		auditID := store.audits[index]
		if !errors.Is(failure, errStepFailed) || !strings.Contains(failure.Error(), auditID) ||
			errors.Is(failure, errReleaseFailed) != (auditID == "audit-c") {
			t.Fatalf("failure %d = %v, want the step failure of %s and its release failure only for audit-c",
				index, failure, auditID)
		}
	}
}

// passCountingStore reports each claim attempt of the Run loop.
type passCountingStore struct {
	*claimScript
	passes chan struct{}
}

func (s *passCountingStore) Claim(ctx context.Context, params auditstore.ClaimParams) ([]auditstore.ControllerClaim, error) {
	s.passes <- struct{}{}
	return s.claimScript.Claim(ctx, params)
}

// waitClock records every wait the Run loop asks for and ends it only when
// the test fires it.
type waitClock struct {
	waits chan time.Duration
	fire  chan time.Time
}

func (*waitClock) Now() time.Time { return time.Date(2026, 10, 24, 0, 0, 0, 0, time.UTC) }

func (c *waitClock) After(duration time.Duration) <-chan time.Time {
	c.waits <- duration
	return c.fire
}

// The Audit Run loop keeps its own pacing: a pass that changed something
// without failing is followed at once by the next pass, and only a pass
// without work waits for the poll interval.
func TestControllerRunPassesAgainAfterWorkAndPollsOtherwise(t *testing.T) {
	harness := newClaimHarness()
	harness.store.snapshot = commonPausedSnapshot(t)
	harness.store.snapshot.Audit.State = auditstore.AuditActive
	deadline := time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC)
	harness.store.snapshot.Audit.DeadlineAt = &deadline
	harness.store.remaining = 1
	store := &passCountingStore{claimScript: harness.store, passes: make(chan struct{})}
	clock := &waitClock{waits: make(chan time.Duration, 1), fire: make(chan time.Time)}
	harness.controller.store = store
	harness.controller.options.Clock = clock
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { done <- harness.controller.Run(ctx) }()

	<-store.passes // The first pass pauses at the pinned deadline: work.
	select {
	case <-store.passes:
	case wait := <-clock.waits:
		t.Fatalf("Run waited %s after a pass that changed the Audit", wait)
	}
	// The scripted next claim is empty: the next pass waits.
	select {
	case wait := <-clock.waits:
		if wait != harness.controller.options.PollInterval {
			t.Fatalf("Run waited %s, want the poll interval %s", wait, harness.controller.options.PollInterval)
		}
	case <-store.passes:
		t.Fatal("Run passed again at once after a pass without work")
	}
	clock.fire <- time.Time{}
	<-store.passes
	<-clock.waits
	cancel()
	if err := <-done; err != nil {
		t.Fatalf("Run after cancellation = %v", err)
	}
}

// claimScript scripts the claim seam only. Unused owner calls panic through
// the nil embedded interface; real Task/transition validation is tested on PG.
type claimScript struct {
	Store
	mu        sync.Mutex
	held      bool
	remaining int
	snapshot  auditstore.ReconcileSnapshot
}

func (s *claimScript) Claim(_ context.Context, p auditstore.ClaimParams) ([]auditstore.ControllerClaim, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.held || s.remaining == 0 {
		return nil, nil
	}
	if s.remaining > 0 {
		s.remaining--
	}
	s.held = true
	return []auditstore.ControllerClaim{{AuditID: "audit-script", HolderID: p.HolderID, Epoch: 1}}, nil
}
func (s *claimScript) ReleaseClaim(_ context.Context, _ auditstore.ControllerClaim) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.held = false
	return nil
}
func (s *claimScript) GetReconcileSnapshot(context.Context, auditstore.ControllerClaim) (auditstore.ReconcileSnapshot, error) {
	return s.snapshot, nil
}
func (s *claimScript) TransitionClaimed(_ context.Context, p auditstore.ClaimedTransitionParams) (auditstore.Audit, error) {
	if p.ExpectedState != s.snapshot.Audit.State || p.ExpectedRevision != s.snapshot.Audit.Revision || p.TargetState != auditstore.AuditPaused {
		return auditstore.Audit{}, auditstore.ErrPrecondition
	}
	s.snapshot.Audit.State = p.TargetState
	s.snapshot.Audit.Revision++
	return s.snapshot.Audit, nil
}
func newClaimHarness() *struct {
	store      *claimScript
	controller *Controller
} {
	store := &claimScript{remaining: -1}
	return &struct {
		store      *claimScript
		controller *Controller
	}{store, &Controller{store: store, drainLimit: maxDrainMutationsPerStep, options: Options{HolderID: "unit-controller", OperationTimeout: time.Second, ClaimLease: 5 * time.Second, PollInterval: time.Second, ClaimBatch: 8, Clock: realClock{}, Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}}}
}
func commonPausedSnapshot(t *testing.T) auditstore.ReconcileSnapshot {
	t.Helper()
	root := configtest.CopyWithPolicies(t, "../config/testdata/valid")
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	program := auditcontract.Program{Schema: "contractor.audit.program.v1", Inputs: map[string]auditcontract.ProgramInput{"source": {Required: true, MediaTypes: []string{"text/plain"}}}, Workflows: map[string]auditcontract.ProgramWorkflow{"inspect": {Ref: "artifact-copy@1", Inputs: map[string]auditcontract.InputMapping{"source": {Source: "audit-input", Name: "source"}}, Parameters: map[string]string{}, Outputs: map[string]string{"material": "result"}}}, Hypotheses: map[string]auditcontract.HypothesisSeed{}, InitialWork: []auditcontract.InitialWork{{Key: "initial", Workflow: "inspect", Objective: "Inspect selected source"}}, Limits: auditcontract.ProgramLimits{MaxTasks: 1, MaxIterations: 1, MaxTasksPerIteration: 1, BatchSize: 1, MaxSubmittedRuns: 1, MaxAttempts: 1, DeadlineSeconds: 60, MaxEvidenceBytes: 1024, MaxHypotheses: 1, MaxHypothesisRevisions: 1, MaxHypothesisBytes: 1024}, Policy: auditcontract.OperationPolicy{ActiveOperations: "prohibited"}}
	data, _ := json.Marshal(program)
	pin, err := auditprofile.PinProgram(resolved.Selector{ID: "claim-script", Version: "1"}, data, catalog.Workflow)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := json.Marshal(pin)
	if err != nil {
		t.Fatal(err)
	}
	return auditstore.ReconcileSnapshot{Audit: auditstore.Audit{AuditID: "audit-script", State: auditstore.AuditPaused, BaselineSnapshot: json.RawMessage(`{}`), ProfileSnapshot: snapshot}}
}
