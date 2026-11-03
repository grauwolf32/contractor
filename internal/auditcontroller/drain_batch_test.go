package auditcontroller

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// These scripts test the batching seam. They do not implement an alternate
// Audit store; authoritative claim/state races are covered on PostgreSQL.
type drainScript struct {
	Store
	RunStore
	RunDeleter
	RunNotifier
	observed, failed, requested, deleted []string
	beforeObserve                        func(int) error
	beforeFailure                        func(int) error
	beforeCancel                         func(int) error
	blocked                              bool
}

func (s *drainScript) ObserveTerminal(_ context.Context, p auditstore.ObserveTerminalParams) (auditstore.Execution, error) {
	if s.beforeObserve != nil {
		if err := s.beforeObserve(len(s.observed)); err != nil {
			return auditstore.Execution{}, err
		}
	}
	if p.RunID == "" || p.Generation == "" || p.Sequence < 1 {
		return auditstore.Execution{}, auditstore.ErrInvalid
	}
	s.observed = append(s.observed, p.ExecutionID)
	return auditstore.Execution{}, nil
}
func (s *drainScript) ObserveSubmissionFailure(_ context.Context, p auditstore.ObserveSubmissionFailureParams) (auditstore.Execution, error) {
	if s.beforeFailure != nil {
		if err := s.beforeFailure(len(s.failed)); err != nil {
			return auditstore.Execution{}, err
		}
	}
	s.failed = append(s.failed, p.ExecutionID)
	return auditstore.Execution{}, nil
}
func (s *drainScript) GetRunEventCursor(context.Context, string) (runstore.WorkflowRunEventCursor, error) {
	return runstore.WorkflowRunEventCursor{Generation: "generation-1", Sequence: 1}, nil
}
func (s *drainScript) RequestRunCancellation(_ context.Context, id string, _ runstore.WorkflowRunCancellation) (runstore.WorkflowRun, error) {
	if s.beforeCancel != nil {
		if err := s.beforeCancel(len(s.requested)); err != nil {
			return runstore.WorkflowRun{}, err
		}
	}
	s.requested = append(s.requested, id)
	return runstore.WorkflowRun{RunID: id, State: runstore.RunCancelling}, nil
}
func (s *drainScript) Cancel(string) {}
func (s *drainScript) NextLiveRunForDeletion(context.Context, auditstore.ControllerClaim) (string, bool, error) {
	if len(s.deleted) == 5 {
		return "", false, nil
	}
	return fmt.Sprintf("run-%d", len(s.deleted)), true, nil
}
func (s *drainScript) DeleteReleasedTerminalRun(_ context.Context, _ string, id string) error {
	if s.blocked && len(s.deleted) == 2 {
		return &rundeletion.NotDeletableError{}
	}
	s.deleted = append(s.deleted, id)
	return nil
}
func drainFixture(total int, state runstore.WorkflowRunState) (*Controller, *drainScript, auditstore.ReconcileSnapshot) {
	s := &drainScript{}
	c := &Controller{store: s, runs: s, deleter: s, notifier: s, drainLimit: maxDrainMutationsPerStep, options: Options{Clock: realClock{}}}
	snapshot := auditstore.ReconcileSnapshot{Audit: auditstore.Audit{OwnerID: "owner"}, RunStates: map[string]runstore.WorkflowRunState{}}
	for i := range total {
		id := fmt.Sprintf("run-%d", i)
		snapshot.Executions = append(snapshot.Executions, auditstore.Execution{ExecutionID: fmt.Sprintf("execution-%d", i), Role: auditstore.ExecutionTask, State: auditstore.ExecutionSubmitted, RunID: &id})
		snapshot.RunStates[id] = state
	}
	return c, s, snapshot
}
func TestDrainBoundStaysWithinTheSnapshotWindow(t *testing.T) {
	if maxDrainMutationsPerStep < 1 || maxDrainMutationsPerStep > auditstore.MaxReconcileRows {
		t.Fatal("drain bound exceeds its snapshot")
	}
	c, _, _ := drainFixture(0, "")
	if c.drainLimit != maxDrainMutationsPerStep {
		t.Fatal("wrong drain bound")
	}
}
func TestControllerObservesTerminalChildrenInBoundedBatches(t *testing.T) {
	total := 2*maxDrainMutationsPerStep + 6
	c, s, snapshot := drainFixture(total, runstore.RunSucceeded)
	for _, want := range []int{maxDrainMutationsPerStep, 2 * maxDrainMutationsPerStep, total} {
		changed, err := c.observeTerminalBatch(t.Context(), auditstore.ControllerClaim{}, snapshot)
		if !changed || err != nil || len(s.observed) != want {
			t.Fatalf("observed %d, changed=%t error=%v; want %d", len(s.observed), changed, err, want)
		}
		snapshot.Executions = snapshot.Executions[min(maxDrainMutationsPerStep, len(snapshot.Executions)):]
	}
	changed, err := c.observeTerminalBatch(t.Context(), auditstore.ControllerClaim{}, snapshot)
	if changed || err != nil {
		t.Fatalf("empty batch=%t,%v", changed, err)
	}
}
func TestControllerDrainBatchStopsAtALostClaim(t *testing.T) {
	c, s, snapshot := drainFixture(12, runstore.RunFailed)
	s.beforeObserve = func(n int) error {
		if n == 5 {
			return auditstore.ErrClaimLost
		}
		return nil
	}
	changed, err := c.observeTerminalBatch(t.Context(), auditstore.ControllerClaim{}, snapshot)
	if !changed || !errors.Is(err, auditstore.ErrClaimLost) || len(s.observed) != 5 {
		t.Fatalf("lost claim batch=%t,%v observations=%v", changed, err, s.observed)
	}
	s.beforeObserve = nil
	snapshot.Executions = snapshot.Executions[5:]
	changed, err = c.observeTerminalBatch(t.Context(), auditstore.ControllerClaim{}, snapshot)
	if !changed || err != nil || len(s.observed) != 12 {
		t.Fatalf("next claim batch=%t,%v observations=%v", changed, err, s.observed)
	}
}
func TestControllerDrainBatchStopsAtAStaleIntent(t *testing.T) {
	c, s, snapshot := drainFixture(10, runstore.RunRunning)
	for i := range snapshot.Executions {
		snapshot.Executions[i].State = auditstore.ExecutionIntent
		snapshot.Executions[i].RunID = nil
	}
	s.beforeFailure = func(n int) error {
		if n == 4 {
			return auditstore.ErrPrecondition
		}
		return nil
	}
	changed, err := c.failUnboundIntentBatch(t.Context(), auditstore.ControllerClaim{}, snapshot)
	if !changed || !errors.Is(err, auditstore.ErrPrecondition) || len(s.failed) != 4 {
		t.Fatalf("stale batch=%t,%v failures=%v", changed, err, s.failed)
	}
	// A fresh owner snapshot reports the racing intent bound; skip it.
	s.beforeFailure = nil
	snapshot.Executions = snapshot.Executions[4:]
	id := "run-racing"
	snapshot.Executions[0].RunID = &id
	snapshot.Executions[0].State = auditstore.ExecutionSubmitted
	changed, err = c.failUnboundIntentBatch(t.Context(), auditstore.ControllerClaim{}, snapshot)
	if !changed || err != nil || len(s.failed) != 9 {
		t.Fatalf("fresh batch=%t,%v failures=%v", changed, err, s.failed)
	}
}
func TestControllerDrainBatchStopsWhenTheStepBudgetEnds(t *testing.T) {
	c, s, snapshot := drainFixture(10, runstore.RunRunning)
	ctx, end := context.WithCancel(t.Context())
	defer end()
	s.beforeCancel = func(n int) error {
		if n == 3 {
			end()
			return ctx.Err()
		}
		return nil
	}
	changed, err := c.requestCancellationBatch(ctx, snapshot)
	if !changed || err != nil || len(s.requested) != 3 {
		t.Fatalf("budget batch=%t,%v requests=%v", changed, err, s.requested)
	}
	s.beforeCancel = nil
	snapshot.Executions = snapshot.Executions[3:]
	changed, err = c.requestCancellationBatch(t.Context(), snapshot)
	if !changed || err != nil || len(s.requested) != 10 {
		t.Fatalf("fresh batch=%t,%v requests=%v", changed, err, s.requested)
	}
}
func TestControllerDeletionBatchStopsAtABlockedRun(t *testing.T) {
	c, s, _ := drainFixture(0, "")
	s.blocked = true
	deleted, drained, err := c.deleteReleasedRunBatch(t.Context(), auditstore.ControllerClaim{}, "owner")
	if !deleted || drained || err != nil || len(s.deleted) != 2 {
		t.Fatalf("blocked deletion=%t,%t,%v rows=%v", deleted, drained, err, s.deleted)
	}
	deleted, drained, err = c.deleteReleasedRunBatch(t.Context(), auditstore.ControllerClaim{}, "owner")
	if deleted || drained || err != nil {
		t.Fatalf("repeated blocked deletion=%t,%t,%v", deleted, drained, err)
	}
	s.blocked = false
	deleted, drained, err = c.deleteReleasedRunBatch(t.Context(), auditstore.ControllerClaim{}, "owner")
	if !deleted || !drained || err != nil || len(s.deleted) != 5 {
		t.Fatalf("released deletion=%t,%t,%v rows=%v", deleted, drained, err, s.deleted)
	}
}

type fixedDrainClock struct{ now time.Time }

func (c fixedDrainClock) Now() time.Time                       { return c.now }
func (c fixedDrainClock) After(time.Duration) <-chan time.Time { return make(chan time.Time) }
func TestControllerLogsADeletionStepBeyondTheOperationBudget(t *testing.T) {
	for _, elapsed := range []time.Duration{time.Millisecond, 2 * time.Second} {
		var logs bytes.Buffer
		started := time.Date(2026, 11, 3, 0, 0, 0, 0, time.UTC)
		c := &Controller{options: Options{Clock: fixedDrainClock{started.Add(elapsed)}, OperationTimeout: time.Second, Logger: slog.New(slog.NewTextHandler(&logs, nil))}}
		c.reportLongDeletionStep(auditstore.ControllerClaim{AuditID: "audit-log"}, started)
		want := elapsed > time.Second
		if strings.Contains(logs.String(), "outlasted") != want || want && !strings.Contains(logs.String(), "audit_id=audit-log") {
			t.Fatalf("elapsed %s: %q", elapsed, logs.String())
		}
	}
}
