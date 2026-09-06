package auditcontroller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestControllerDispatchesThroughWindowAndOnlyObservesTerminal(t *testing.T) {
	harness := newControllerHarness(t, 4, 2)
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("activate round = (%t, %v)", worked, err)
	}
	for range 2 {
		if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
			t.Fatalf("dispatch initial child = (%t, %v)", worked, err)
		}
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || worked {
		t.Fatalf("dispatch beyond window = (%t, %v)", worked, err)
	}
	if got := harness.store.executionCount(); got != 2 {
		t.Fatalf("initial execution count = %d, want 2", got)
	}
	if got := harness.store.maximumOutstanding(); got != 2 {
		t.Fatalf("maximum outstanding = %d, want 2", got)
	}

	harness.finishOldest(t, runstore.RunSucceeded)
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("observe first terminal Run = (%t, %v)", worked, err)
	}
	if state := harness.store.itemState(0); state != auditstore.ItemCollecting {
		t.Fatalf("terminal observation settled item: state = %q", state)
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("dispatch replacement child = (%t, %v)", worked, err)
	}
	harness.finishOldest(t, runstore.RunFailed)
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("observe second terminal Run = (%t, %v)", worked, err)
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("dispatch final child = (%t, %v)", worked, err)
	}
	if got := harness.store.executionCount(); got != 4 {
		t.Fatalf("execution count = %d, want one per immutable item", got)
	}
	if got := harness.creator.createdCount(); got != 4 {
		t.Fatalf("Run count = %d, want 4", got)
	}
	if got := harness.store.maximumOutstanding(); got != 2 {
		t.Fatalf("maximum outstanding after refill = %d, want 2", got)
	}
}

func TestControllerReplaysIntentAfterSubmissionFailure(t *testing.T) {
	harness := newControllerHarness(t, 1, 1)
	harness.creator.failNext.Store(true)
	if _, err := harness.controller.RunOnce(harness.ctx); err != nil { // accepted -> executing
		t.Fatal(err)
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err == nil || worked {
		t.Fatalf("transient submission = (%t, %v), want retained intent and error", worked, err)
	}
	if got := harness.store.executionCount(); got != 1 {
		t.Fatalf("intent count after transient failure = %d", got)
	}
	if got := harness.creator.createdCount(); got != 0 {
		t.Fatalf("Run committed during failed call = %d", got)
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("intent replay = (%t, %v)", worked, err)
	}
	if got := harness.store.executionCount(); got != 1 {
		t.Fatalf("intent replay duplicated execution: %d", got)
	}
	if got := harness.creator.createdCount(); got != 1 {
		t.Fatalf("intent replay Run count = %d", got)
	}
	harness.finishOldest(t, runstore.RunSucceeded)
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("terminal recovery = (%t, %v)", worked, err)
	}
	if state := harness.store.itemState(0); state != auditstore.ItemCollecting {
		t.Fatalf("recovered item state = %q", state)
	}
}

func TestControllerCancellationClosesDispatchAndCancelsBoundRun(t *testing.T) {
	harness := newControllerHarness(t, 2, 2)
	_, _ = harness.controller.RunOnce(harness.ctx)
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("initial dispatch = (%t, %v)", worked, err)
	}
	harness.store.cancelAudit()
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("cancel child = (%t, %v)", worked, err)
	}
	if got := harness.creator.createdCount(); got != 1 {
		t.Fatalf("cancellation dispatched another Run: %d", got)
	}
	if got := harness.runs.cancelCount(); got != 1 {
		t.Fatalf("child cancellation count = %d, want 1", got)
	}
	if got := harness.notifier.cancelCount(); got != 1 {
		t.Fatalf("Scheduler cancellation hints = %d, want 1", got)
	}
}

func TestControllerDeadlineClosesDispatchBeforeRoundOrRunCreation(t *testing.T) {
	harness := newControllerHarness(t, 2, 2)
	harness.store.expireAudit()
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("deadline fence = (%t, %v)", worked, err)
	}
	audit := harness.store.auditSnapshot()
	if audit.State != auditstore.AuditFinalizing || audit.Dispatch != auditstore.DispatchClosed ||
		audit.StopReason == nil || audit.StopReason.Code != "deadline_exhausted" {
		t.Fatalf("deadline result = %+v", audit)
	}
	if got := harness.creator.createdCount(); got != 0 {
		t.Fatalf("deadline created %d child Runs", got)
	}
}

func TestControllerExpiresPendingReportAcceptance(t *testing.T) {
	harness := newControllerHarness(t, 0, 1)
	harness.store.mu.Lock()
	harness.store.audit.State = auditstore.AuditWaitingReview
	harness.store.audit.Dispatch = auditstore.DispatchClosed
	harness.store.expiredReportReview = true
	harness.store.mu.Unlock()

	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("expire report acceptance = (%t, %v)", worked, err)
	}
	audit := harness.store.auditSnapshot()
	if audit.State != auditstore.AuditFailed || audit.StopReason == nil ||
		audit.StopReason.Code != "report_acceptance_expired" {
		t.Fatalf("expired report Audit = %+v", audit)
	}
}

func TestControllerCollectsClosesBarrierAndFinalizesReport(t *testing.T) {
	harness := newControllerHarness(t, 1, 1)
	collector := &fakeControllerCollector{store: harness.store}
	harness.controller.collector = collector
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("activate round = (%t, %v)", worked, err)
	}
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("dispatch = (%t, %v)", worked, err)
	}
	harness.finishOldest(t, runstore.RunSucceeded)
	for step, name := range []string{
		"observe", "collect", "begin assessment", "close round", "begin finalizing", "commit report",
	} {
		if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
			t.Fatalf("%s at step %d = (%t, %v)", name, step, worked, err)
		}
	}
	audit := harness.store.auditSnapshot()
	if audit.State != auditstore.AuditCompleted || collector.collects.Load() != 1 || collector.finalizes.Load() != 1 {
		t.Fatalf("terminal Audit = %+v, collects=%d finalizes=%d", audit, collector.collects.Load(), collector.finalizes.Load())
	}
}

func TestControllerAcceptsPreparedNextRoundInsteadOfFinalizing(t *testing.T) {
	harness := newControllerHarness(t, 1, 1)
	harness.store.mu.Lock()
	harness.store.round.State = auditstore.RoundClosed
	harness.store.items = nil
	harness.store.executions = nil
	harness.store.audit.Limits.MaxRounds = 2
	harness.store.audit.Limits.MaxItemsTotal = 2
	harness.store.mu.Unlock()

	roundBuilder := &fakeRoundBuilder{}
	harness.controller.roundBuilder = roundBuilder
	if worked, err := harness.controller.RunOnce(harness.ctx); err != nil || !worked {
		t.Fatalf("accept next Round = (%t, %v)", worked, err)
	}
	if roundBuilder.calls.Load() != 1 {
		t.Fatalf("next Round builder calls = %d, want 1", roundBuilder.calls.Load())
	}
	harness.store.mu.Lock()
	defer harness.store.mu.Unlock()
	if harness.store.audit.State != auditstore.AuditActive ||
		harness.store.round.Ordinal != 2 || harness.store.round.State != auditstore.RoundAccepted ||
		len(harness.store.items) != 1 || harness.store.items[0].RoundID != "round-next" {
		t.Fatalf("accepted next Round = (audit=%+v round=%+v items=%+v)",
			harness.store.audit, harness.store.round, harness.store.items)
	}
}

func TestRoleDispositionRetryabilityIsExplicit(t *testing.T) {
	evidenceBudget := "evidence-budget-exhausted"
	for _, test := range []struct {
		name        string
		disposition auditstore.CollectionDisposition
		code        *string
		want        bool
	}{
		{"failed execution", auditstore.CollectionExecutionFailed, nil, true},
		{"missing output", auditstore.CollectionMissingOutput, nil, true},
		{"invalid output", auditstore.CollectionInvalidResult, nil, true},
		{"evidence budget", auditstore.CollectionInvalidResult, &evidenceBudget, false},
		{"cancelled", auditstore.CollectionExecutionCancelled, nil, false},
		{"contract invalid", auditstore.CollectionContractInvalid, nil, false},
		{"accepted", auditstore.CollectionAccepted, nil, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := roleDispositionRetryable(test.disposition, test.code); got != test.want {
				t.Fatalf("role disposition retryable = %t, want %t", got, test.want)
			}
		})
	}
}

type controllerHarness struct {
	ctx        context.Context
	store      *fakeControllerStore
	runs       *fakeControllerRuns
	creator    *fakeControllerCreator
	notifier   *fakeControllerNotifier
	controller *Controller
}

func newControllerHarness(t *testing.T, itemCount, window int) *controllerHarness {
	t.Helper()
	ctx := context.Background()
	store := newFakeControllerStore(t, itemCount, window)
	runs := &fakeControllerRuns{runs: make(map[string]runstore.WorkflowRun), cursors: make(map[string]runstore.WorkflowRunEventCursor)}
	creator := &fakeControllerCreator{store: store, runs: runs}
	notifier := &fakeControllerNotifier{}
	var ids atomic.Int64
	controller, err := New(store, runs, creator, fakeSubmissionBuilder{}, notifier, Options{
		HolderID: "controller-test", ClaimLease: time.Second,
		OperationTimeout: 100 * time.Millisecond, ClaimBatch: 1,
		NewID: func(prefix string) (string, error) {
			return fmt.Sprintf("%s%d", prefix, ids.Add(1)), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return &controllerHarness{ctx: ctx, store: store, runs: runs, creator: creator, notifier: notifier, controller: controller}
}

func (h *controllerHarness) finishOldest(t *testing.T, state runstore.WorkflowRunState) {
	t.Helper()
	runID := h.runs.oldestActive()
	if runID == "" {
		t.Fatal("no active child Run")
	}
	h.runs.finish(runID, state)
}

type fakeSubmissionBuilder struct{}

type fakeRoundBuilder struct{ calls atomic.Int64 }

func (b *fakeRoundBuilder) PrepareNextRound(
	_ context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
) (auditstore.AcceptRoundParams, *auditstore.StopReason, error) {
	b.calls.Add(1)
	manifest := builderExact("round-two", "round-two-r1")
	task := builderExact("task-two", "task-two-r1")
	return auditstore.AcceptRoundParams{
		Claim: claim, ExpectedAuditRevision: snapshot.Audit.Revision,
		PreviousRoundID: snapshot.Round.RoundID, RoundID: "round-next", RoundOrdinal: 2,
		Manifest: manifest,
		Items: []auditstore.MaterializedItem{{
			ItemID: "item-next", ItemKey: "finding-next", Ordinal: 0,
			Kind: "finding-verification", SubjectKey: "subject-next", Task: task,
			WorkflowRole: "check", InitialState: auditstore.ItemReady,
		}},
	}, nil, nil
}

func (fakeSubmissionBuilder) PrepareRole(
	_ context.Context, _ auditstore.ReconcileSnapshot, _ string, _ int,
) (PreparedSubmission, error) {
	return PreparedSubmission{}, ErrInvalidSubmission
}

func (fakeSubmissionBuilder) Prepare(
	_ context.Context, snapshot auditstore.ReconcileSnapshot, item auditstore.Item, attempt int,
) (PreparedSubmission, error) {
	revision := fmt.Sprintf("manifest-%s-%d", item.ItemID, attempt)
	manifest := auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "audit-test", Name: "manifest-" + item.ItemID, Revision: &revision},
		Digest: fakeDigest(item.ItemID), MediaType: "application/json", SizeBytes: 1,
	}
	roundID := item.RoundID
	executionID := fmt.Sprintf("execution-%s-%d", item.ItemID, attempt)
	memberID := fmt.Sprintf("member-%s-%d", item.ItemID, attempt)
	requestDigest := fakeDigest(executionID)
	return PreparedSubmission{
		Intent: auditstore.CreateExecutionIntentParams{
			ExecutionID: executionID, RoundID: &roundID, Role: auditstore.ExecutionCheck,
			WorkflowRole: item.WorkflowRole,
			Manifest:     manifest, SubmissionKey: "submission-" + executionID,
			RequestDigest: requestDigest,
			Members: []auditstore.ExecutionMemberIntent{{
				ExecutionItemID: memberID, ItemID: item.ItemID, BatchOrdinal: 0,
				ItemAttempt: attempt, Task: item.Task, Inputs: []auditstore.ExactArtifact{},
			}},
		},
		Run: runservice.AuditCreateParams{
			ExecutionID: executionID, ExecutionManifest: manifest, RequestDigest: requestDigest,
		},
	}, nil
}

type fakeControllerStore struct {
	mu                  sync.Mutex
	audit               auditstore.Audit
	round               auditstore.Round
	items               []auditstore.Item
	executions          []auditstore.Execution
	members             map[string][]auditstore.ExecutionItem
	held                bool
	epoch               uint64
	window              int
	maxOutstanding      int
	expiredReportReview bool
}

func newFakeControllerStore(t *testing.T, itemCount, window int) *fakeControllerStore {
	t.Helper()
	snapshot, err := config.Load("../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := snapshot.AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	profileJSON, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Hour)
	roundID := "round-test"
	store := &fakeControllerStore{
		audit: auditstore.Audit{
			AuditID: "audit-test", OwnerID: "owner-test", ProjectID: "project-test",
			Profile: auditstore.ProfileIdentity{
				Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest,
			},
			ProfileSnapshot: profileJSON,
			State:           auditstore.AuditActive, Revision: 2, CurrentRoundID: &roundID,
			Dispatch: auditstore.DispatchOpen, DeadlineAt: &deadline,
			Limits: auditstore.Limits{
				MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: itemCount,
				MaxItemsTotal: itemCount, MaxSubmittedRuns: itemCount,
				MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
			},
		},
		round: auditstore.Round{
			RoundID: roundID, AuditID: "audit-test", Ordinal: 1,
			State: auditstore.RoundAccepted, Revision: 1, ExpectedItemCount: itemCount,
		},
		window: window, members: make(map[string][]auditstore.ExecutionItem),
	}
	for index := range itemCount {
		revision := fmt.Sprintf("task-r%d", index)
		store.items = append(store.items, auditstore.Item{
			ItemID: fmt.Sprintf("item-%d", index), AuditID: "audit-test", RoundID: roundID,
			ItemKey: fmt.Sprintf("check-%d", index), Ordinal: index, Kind: "checklist",
			SubjectKey: fmt.Sprintf("subject-%d", index), WorkflowRole: "check",
			State: auditstore.ItemReady,
			Task: auditstore.ExactArtifact{
				Ref:    contracts.ArtifactRef{Namespace: "audit-test", Name: fmt.Sprintf("task-%d", index), Revision: &revision},
				Digest: fakeDigest(fmt.Sprintf("task-%d", index)),
			},
		})
	}
	return store
}

func (s *fakeControllerStore) Claim(_ context.Context, params auditstore.ClaimParams) ([]auditstore.ControllerClaim, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.held || s.audit.State.Terminal() {
		return nil, nil
	}
	s.held = true
	s.epoch++
	now := time.Now()
	return []auditstore.ControllerClaim{{
		AuditID: s.audit.AuditID, HolderID: params.HolderID, Epoch: s.epoch,
		ClaimedAt: now, ExpiresAt: now.Add(params.Lease),
	}}, nil
}

func (s *fakeControllerStore) ReleaseClaim(_ context.Context, claim auditstore.ControllerClaim) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.held || claim.Epoch != s.epoch {
		return auditstore.ErrClaimLost
	}
	s.held = false
	return nil
}

func (s *fakeControllerStore) ExpireReportReview(
	_ context.Context, claim auditstore.ControllerClaim, expectedRevision uint64,
) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.expiredReportReview {
		return false, nil
	}
	if !s.held || claim.Epoch != s.epoch || s.audit.Revision != expectedRevision ||
		s.audit.State != auditstore.AuditWaitingReview {
		return false, auditstore.ErrPrecondition
	}
	s.expiredReportReview = false
	s.audit.State = auditstore.AuditFailed
	s.audit.Revision++
	s.audit.StopReason = &auditstore.StopReason{
		Code: "report_acceptance_expired", Message: "The report approval expired.",
	}
	return true, nil
}

func (s *fakeControllerStore) GetReconcileSnapshot(_ context.Context, claim auditstore.ControllerClaim) (auditstore.ReconcileSnapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.held || claim.Epoch != s.epoch {
		return auditstore.ReconcileSnapshot{}, auditstore.ErrClaimLost
	}
	round := s.round
	return auditstore.ReconcileSnapshot{
		Audit: s.audit, Round: &round,
		Items:      append([]auditstore.Item(nil), s.items...),
		Executions: append([]auditstore.Execution(nil), s.executions...),
	}, nil
}

func (s *fakeControllerStore) TransitionRound(_ context.Context, params auditstore.RoundTransitionParams) (auditstore.Round, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if params.Claim.Epoch != s.epoch || s.round.State != params.ExpectedState || s.round.Revision != params.ExpectedRevision {
		return auditstore.Round{}, auditstore.ErrPrecondition
	}
	s.round.State = params.TargetState
	s.round.Revision++
	s.audit.Revision++
	return s.round, nil
}

func (s *fakeControllerStore) TransitionClaimed(_ context.Context, params auditstore.ClaimedTransitionParams) (auditstore.Audit, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if params.Claim.Epoch != s.epoch || s.audit.State != params.ExpectedState || s.audit.Revision != params.ExpectedRevision {
		return auditstore.Audit{}, auditstore.ErrPrecondition
	}
	s.audit.State = params.TargetState
	s.audit.Revision++
	s.audit.StopReason = params.Reason
	if params.TargetState == auditstore.AuditFinalizing || params.TargetState == auditstore.AuditCancelling {
		s.audit.Dispatch = auditstore.DispatchClosed
	}
	return s.audit, nil
}

func (s *fakeControllerStore) CreateExecutionIntent(_ context.Context, params auditstore.CreateExecutionIntentParams) (auditstore.Execution, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, existing := range s.executions {
		if existing.SubmissionKey == params.SubmissionKey {
			return existing, false, nil
		}
	}
	if params.Claim.Epoch != s.epoch || s.audit.State != auditstore.AuditActive ||
		s.audit.Dispatch != auditstore.DispatchOpen || s.audit.OutstandingRunCount >= s.window ||
		s.audit.ReservedRunCount >= s.audit.Limits.MaxSubmittedRuns || len(params.Members) != 1 {
		return auditstore.Execution{}, false, auditstore.ErrPrecondition
	}
	member := params.Members[0]
	itemIndex := s.findItem(member.ItemID)
	if itemIndex < 0 || s.items[itemIndex].State != auditstore.ItemReady {
		return auditstore.Execution{}, false, auditstore.ErrPrecondition
	}
	execution := auditstore.Execution{
		ExecutionID: params.ExecutionID, AuditID: s.audit.AuditID, RoundID: params.RoundID,
		Role: params.Role, WorkflowRole: params.WorkflowRole, Manifest: params.Manifest, SubmissionKey: params.SubmissionKey,
		RequestDigest: params.RequestDigest, State: auditstore.ExecutionIntent,
	}
	s.executions = append(s.executions, execution)
	s.members[execution.ExecutionID] = []auditstore.ExecutionItem{{
		ExecutionItemID: member.ExecutionItemID, ExecutionID: execution.ExecutionID,
		AuditID: s.audit.AuditID, RoundID: s.round.RoundID, ItemID: member.ItemID,
		BatchOrdinal: member.BatchOrdinal, ItemAttempt: member.ItemAttempt,
		Task: member.Task, Inputs: append([]auditstore.ExactArtifact(nil), member.Inputs...),
		State: auditstore.ItemSubmitted,
	}}
	s.items[itemIndex].State = auditstore.ItemSubmitted
	s.items[itemIndex].LastExecutionItemID = &member.ExecutionItemID
	s.audit.ReservedRunCount++
	s.audit.OutstandingRunCount++
	if s.audit.OutstandingRunCount > s.maxOutstanding {
		s.maxOutstanding = s.audit.OutstandingRunCount
	}
	s.audit.Revision++
	return execution, true, nil
}

func (s *fakeControllerStore) NextItemAttempt(_ context.Context, claim auditstore.ControllerClaim, itemID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if claim.Epoch != s.epoch {
		return 0, auditstore.ErrClaimLost
	}
	maximum := 0
	for _, members := range s.members {
		for _, member := range members {
			if member.ItemID == itemID && member.ItemAttempt > maximum {
				maximum = member.ItemAttempt
			}
		}
	}
	return maximum + 1, nil
}

func (s *fakeControllerStore) ListExecutionItems(_ context.Context, executionID string) ([]auditstore.ExecutionItem, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]auditstore.ExecutionItem(nil), s.members[executionID]...), nil
}

func (s *fakeControllerStore) ObserveTerminal(_ context.Context, params auditstore.ObserveTerminalParams) (auditstore.Execution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	index := s.findExecution(params.ExecutionID)
	if index < 0 || s.executions[index].State != auditstore.ExecutionSubmitted {
		return auditstore.Execution{}, auditstore.ErrPrecondition
	}
	outcome := auditstore.TerminalOutcome(s.executionRunState(params.ExecutionID))
	s.executions[index].State = auditstore.ExecutionCollecting
	s.executions[index].TerminalOutcome = &outcome
	s.executions[index].TerminalRunGeneration = &params.Generation
	sequence := params.Sequence
	s.executions[index].TerminalRunSequence = &sequence
	for memberIndex := range s.members[params.ExecutionID] {
		s.members[params.ExecutionID][memberIndex].State = auditstore.ItemCollecting
		itemIndex := s.findItem(s.members[params.ExecutionID][memberIndex].ItemID)
		s.items[itemIndex].State = auditstore.ItemCollecting
	}
	s.audit.OutstandingRunCount--
	s.audit.Revision++
	return s.executions[index], nil
}

func (s *fakeControllerStore) ObserveSubmissionFailure(_ context.Context, params auditstore.ObserveSubmissionFailureParams) (auditstore.Execution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	index := s.findExecution(params.ExecutionID)
	if index < 0 || s.executions[index].State != auditstore.ExecutionIntent {
		return auditstore.Execution{}, auditstore.ErrPrecondition
	}
	outcome := auditstore.TerminalSubmissionFailed
	s.executions[index].State = auditstore.ExecutionCollecting
	s.executions[index].TerminalOutcome = &outcome
	for _, member := range s.members[params.ExecutionID] {
		itemIndex := s.findItem(member.ItemID)
		s.items[itemIndex].State = auditstore.ItemCollecting
	}
	s.audit.OutstandingRunCount--
	return s.executions[index], nil
}

func (s *fakeControllerStore) SettleUndispatched(
	_ context.Context, claim auditstore.ControllerClaim, limit int,
) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.held || claim.Epoch != s.epoch {
		return 0, auditstore.ErrClaimLost
	}
	if s.audit.Dispatch != auditstore.DispatchClosed {
		return 0, auditstore.ErrPrecondition
	}
	settled := 0
	kept := s.items[:0]
	for _, item := range s.items {
		if settled < limit && (item.State == auditstore.ItemPending ||
			item.State == auditstore.ItemAwaitingReview || item.State == auditstore.ItemReady) {
			settled++
			continue
		}
		kept = append(kept, item)
	}
	s.items = kept
	if settled != 0 {
		s.audit.Revision++
	}
	return settled, nil
}

func (s *fakeControllerStore) AcceptNextRound(
	_ context.Context, params auditstore.AcceptRoundParams,
) (auditstore.Round, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.held || params.Claim.Epoch != s.epoch ||
		s.audit.State != auditstore.AuditActive || s.round.State != auditstore.RoundClosed ||
		s.audit.Revision != params.ExpectedAuditRevision ||
		s.round.RoundID != params.PreviousRoundID {
		return auditstore.Round{}, false, auditstore.ErrPrecondition
	}
	s.round = auditstore.Round{
		RoundID: params.RoundID, AuditID: s.audit.AuditID, Ordinal: params.RoundOrdinal,
		State: auditstore.RoundAccepted, Revision: 1, Manifest: params.Manifest,
		ExpectedItemCount: len(params.Items),
	}
	s.audit.CurrentRoundID = &s.round.RoundID
	s.audit.Revision++
	s.items = make([]auditstore.Item, len(params.Items))
	for index, item := range params.Items {
		s.items[index] = auditstore.Item{
			ItemID: item.ItemID, AuditID: s.audit.AuditID, RoundID: params.RoundID,
			ItemKey: item.ItemKey, Ordinal: item.Ordinal, Kind: item.Kind,
			SubjectKey: item.SubjectKey, Task: item.Task, WorkflowRole: item.WorkflowRole,
			State: item.InitialState,
		}
	}
	return s.round, true, nil
}

func (s *fakeControllerStore) ReleaseDispatchHold(
	_ context.Context, claim auditstore.ControllerClaim,
) (auditstore.Audit, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.held || claim.Epoch != s.epoch {
		return auditstore.Audit{}, false, auditstore.ErrClaimLost
	}
	if s.audit.Hold != auditstore.HoldHeld {
		return s.audit, false, nil
	}
	for _, execution := range s.executions {
		if execution.State == auditstore.ExecutionIntent {
			return s.audit, false, nil
		}
	}
	s.audit.Hold = auditstore.HoldReleased
	s.audit.Revision++
	return s.audit, true, nil
}

func (s *fakeControllerStore) NextLiveRunForDeletion(
	_ context.Context, claim auditstore.ControllerClaim,
) (string, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.held || claim.Epoch != s.epoch {
		return "", false, auditstore.ErrClaimLost
	}
	return "", false, nil
}

func (s *fakeControllerStore) PurgeClaimed(
	_ context.Context, claim auditstore.ControllerClaim, _ string,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.held || claim.Epoch != s.epoch {
		return auditstore.ErrClaimLost
	}
	s.audit.State = auditstore.AuditCompleted
	s.held = false
	return nil
}

func (s *fakeControllerStore) findItem(id string) int {
	for index := range s.items {
		if s.items[index].ItemID == id {
			return index
		}
	}
	return -1
}

func (s *fakeControllerStore) findExecution(id string) int {
	for index := range s.executions {
		if s.executions[index].ExecutionID == id {
			return index
		}
	}
	return -1
}

func (s *fakeControllerStore) executionRunState(executionID string) string {
	index := s.findExecution(executionID)
	if index < 0 || s.executions[index].RunID == nil {
		return "failed"
	}
	return "succeeded"
}

func (s *fakeControllerStore) bind(executionID, runID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	index := s.findExecution(executionID)
	if index < 0 {
		return errors.New("execution missing")
	}
	if s.executions[index].RunID != nil {
		return nil
	}
	s.executions[index].RunID = &runID
	s.executions[index].State = auditstore.ExecutionSubmitted
	s.audit.SubmittedRunCount++
	return nil
}

func (s *fakeControllerStore) cancelAudit() {
	s.mu.Lock()
	s.audit.State = auditstore.AuditCancelling
	s.audit.Dispatch = auditstore.DispatchClosed
	s.audit.Revision++
	s.mu.Unlock()
}

func (s *fakeControllerStore) expireAudit() {
	s.mu.Lock()
	deadline := time.Now().Add(-time.Second)
	s.audit.DeadlineAt = &deadline
	s.mu.Unlock()
}

func (s *fakeControllerStore) auditSnapshot() auditstore.Audit {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.audit
}

func (s *fakeControllerStore) executionCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.executions)
}

func (s *fakeControllerStore) maximumOutstanding() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxOutstanding
}

func (s *fakeControllerStore) itemState(index int) auditstore.ItemState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.items[index].State
}

type fakeControllerRuns struct {
	mu            sync.Mutex
	runs          map[string]runstore.WorkflowRun
	cursors       map[string]runstore.WorkflowRunEventCursor
	order         []string
	cancellations int
}

func (r *fakeControllerRuns) GetRun(_ context.Context, runID string) (runstore.WorkflowRun, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	run, exists := r.runs[runID]
	if !exists {
		return runstore.WorkflowRun{}, runstore.ErrNotFound
	}
	return run, nil
}

func (r *fakeControllerRuns) GetRunEventCursor(_ context.Context, runID string) (runstore.WorkflowRunEventCursor, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cursors[runID], nil
}

func (r *fakeControllerRuns) RequestRunCancellation(_ context.Context, runID string, _ runstore.WorkflowRunCancellation) (runstore.WorkflowRun, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	run := r.runs[runID]
	if run.State == runstore.RunInitializing || run.State == runstore.RunRunning {
		run.State = runstore.RunCancelling
		r.runs[runID] = run
		r.cancellations++
	}
	return run, nil
}

func (r *fakeControllerRuns) DeleteReleasedTerminalRun(
	_ context.Context, _ string, runID string,
) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	run, exists := r.runs[runID]
	if !exists {
		return runstore.ErrNotFound
	}
	if !runstore.RunLifecycleTerminal.Includes(run.State) {
		return &runstore.RunNotDeletableError{RunID: runID, Reason: runstore.RunNotTerminal}
	}
	delete(r.runs, runID)
	delete(r.cursors, runID)
	return nil
}

func (r *fakeControllerRuns) finish(runID string, state runstore.WorkflowRunState) {
	r.mu.Lock()
	run := r.runs[runID]
	run.State = state
	r.runs[runID] = run
	r.cursors[runID] = runstore.WorkflowRunEventCursor{Generation: "generation-1", Sequence: 3}
	r.mu.Unlock()
}

func (r *fakeControllerRuns) oldestActive() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, runID := range r.order {
		state := r.runs[runID].State
		if state == runstore.RunRunning || state == runstore.RunInitializing {
			return runID
		}
	}
	return ""
}

func (r *fakeControllerRuns) cancelCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cancellations
}

type fakeControllerCreator struct {
	store    *fakeControllerStore
	runs     *fakeControllerRuns
	failNext atomic.Bool
	created  atomic.Int64
}

func (c *fakeControllerCreator) CreateAudit(_ context.Context, params runservice.AuditCreateParams) (runservice.CreateResult, error) {
	if c.failNext.CompareAndSwap(true, false) {
		return runservice.CreateResult{}, errors.New("temporary database outage")
	}
	c.store.mu.Lock()
	index := c.store.findExecution(params.ExecutionID)
	if index < 0 {
		c.store.mu.Unlock()
		return runservice.CreateResult{}, errors.New("intent missing")
	}
	if c.store.executions[index].RunID != nil {
		runID := *c.store.executions[index].RunID
		c.store.mu.Unlock()
		run, err := c.runs.GetRun(context.Background(), runID)
		return runservice.CreateResult{Run: run, Replayed: true}, err
	}
	c.store.mu.Unlock()
	runID, err := params.NewRunID()
	if err != nil {
		return runservice.CreateResult{}, err
	}
	run := runstore.WorkflowRun{RunID: runID, State: runstore.RunRunning}
	c.runs.mu.Lock()
	c.runs.runs[runID] = run
	c.runs.order = append(c.runs.order, runID)
	c.runs.mu.Unlock()
	if err := c.store.bind(params.ExecutionID, runID); err != nil {
		return runservice.CreateResult{}, err
	}
	c.created.Add(1)
	return runservice.CreateResult{Run: run, Created: true}, nil
}

func (c *fakeControllerCreator) createdCount() int { return int(c.created.Load()) }

type fakeControllerNotifier struct {
	wakes   atomic.Int64
	cancels atomic.Int64
}

func (n *fakeControllerNotifier) Wake()            { n.wakes.Add(1) }
func (n *fakeControllerNotifier) Cancel(string)    { n.cancels.Add(1) }
func (n *fakeControllerNotifier) cancelCount() int { return int(n.cancels.Load()) }

type fakeControllerCollector struct {
	store     *fakeControllerStore
	collects  atomic.Int64
	finalizes atomic.Int64
}

func (c *fakeControllerCollector) Collect(
	_ context.Context,
	_ auditstore.ControllerClaim,
	_ auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
) (bool, error) {
	c.store.mu.Lock()
	defer c.store.mu.Unlock()
	index := c.store.findExecution(execution.ExecutionID)
	if index < 0 || c.store.executions[index].State != auditstore.ExecutionCollecting {
		return false, auditstore.ErrPrecondition
	}
	for _, member := range c.store.members[execution.ExecutionID] {
		itemIndex := c.store.findItem(member.ItemID)
		if itemIndex >= 0 {
			c.store.items = append(c.store.items[:itemIndex], c.store.items[itemIndex+1:]...)
		}
	}
	c.store.executions = append(c.store.executions[:index], c.store.executions[index+1:]...)
	c.store.audit.Revision++
	c.collects.Add(1)
	return true, nil
}

func (c *fakeControllerCollector) Finalize(
	_ context.Context,
	_ auditstore.ControllerClaim,
	_ auditstore.ReconcileSnapshot,
) (bool, error) {
	c.store.mu.Lock()
	defer c.store.mu.Unlock()
	if c.store.audit.State != auditstore.AuditFinalizing || c.store.round.State != auditstore.RoundClosed {
		return false, nil
	}
	c.store.audit.State = auditstore.AuditCompleted
	c.store.audit.Revision++
	c.finalizes.Add(1)
	return true, nil
}

func fakeDigest(value string) string {
	result := ""
	for len(result) < 64 {
		result += fmt.Sprintf("%x", value)
		if value == "" {
			result += "0"
		}
	}
	return "sha256:" + result[:64]
}
