package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// Replacement must fence the commit itself, before any renewal or process
// liveness probe notices it. The original report/cleanup context is still live.
func TestPostgresReplacedClaimCannotCommitBeforeRenewal(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	fixture := createFinalizingFixture(t, ctx, pool)
	store := &claimLossReportStore{PostgresStore: fixture.store, reports: telemetry.NewRepository(pool), started: make(chan context.Context, 1), resume: make(chan struct{})}
	s := newLaneTestScheduler(t, newLaneTestStore(), newMutableSchedulerSettings(2, time.Second), time.Second)
	s.store, s.options.Reports, s.persistence = store, store, fixture.persistence
	s.options.Clock = newRenewalClock(s.options.ClaimDuration)
	lane, stop := context.WithCancel(ctx)
	defer stop()
	done := make(chan error, 1)
	go func() { _, err := s.progressOneClaim(lane, resolved.LaneClassModel, nil); done <- err }()
	defer store.unblock()
	awaitValue(t, ctx, store.started, "report before commit")
	stealRunClaim(t, ctx, pool, "run-1")
	store.unblock() // Do not release the renewal timer.
	if err := awaitValue(t, ctx, done, "old lane completion"); !errors.Is(err, ErrClaimLost) {
		t.Fatalf("old lane commit = %v, want claim loss", err)
	}
	run, err := fixture.store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunRunning || run.SchedulerClaim == nil || run.SchedulerClaim.ClaimID != "competing-lane" {
		t.Fatalf("old lane changed replacement ownership: state=%s claim=%+v err=%v", run.State, run.SchedulerClaim, err)
	}
	stage, err := fixture.store.GetStageExecution(ctx, fixture.executionID)
	if err != nil || stage.State != runstore.StageFinalizing {
		t.Fatalf("old lane progressed the Stage: %s %v", stage.State, err)
	}
}

// All commit points use the same row-locked ownership boundary, including
// detached cancellation cleanup. No missing or wrong-Run scope is an escape.
func TestPostgresLifecycleCommitsRequireExactClaim(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	f := createFinalizingFixture(t, ctx, pool)
	old := fixtureClaim(t, ctx, pool, "run-1")
	state := runstore.RunRunning
	result := ResultProgression{
		RunID: "run-1", StageExecutionID: f.executionID, Result: f.result,
		WorkflowOutputs: f.workflow.Stages["copy"].WorkflowOutputs, OutputContracts: f.workflow.Outputs,
		Progression: StageProgression{
			Decision: runstore.RecordStageTransitionDecisionParams{
				RunID: "run-1", SourceExecutionID: f.executionID, Action: runstore.StageTransitionSucceed,
			},
			TerminalRunState: runstore.RunSucceeded, RunReason: runstore.Reason{Code: "workflow_succeeded"},
		},
	}
	term := TerminationProgression{
		RunID: "run-1", StageExecutionID: f.executionID,
		Progression: StageProgression{
			Decision: runstore.RecordStageTransitionDecisionParams{
				RunID: "run-1", SourceExecutionID: f.executionID, Action: runstore.StageTransitionFail,
			},
			TerminalRunState: runstore.RunFailed, RunReason: runstore.Reason{Code: "stage_failed"},
		},
	}
	stealRunClaim(t, ctx, pool, "run-1")
	calls := map[string]func(context.Context) error{
		"admission": func(c context.Context) error { _, e := f.persistence.AdmitStage(c, "run-1", f.executionID); return e },
		"candidate": func(c context.Context) error {
			return f.persistence.EnterFinalizingWithResult(c, runstore.EnterFinalizingParams{
				StageExecutionID: f.executionID, ResultSchemaVersion: contracts.APIVersion, Candidate: f.result,
				FinalizationID: "finalization-fenced", Deadline: time.Now().Add(time.Minute),
				Reason: runstore.Reason{Code: "planner_completed"},
			})
		},
		"abort": func(c context.Context) error {
			return f.persistence.EnterAbortingWithTermination(c, "run-1", runstore.EnterAbortingParams{
				StageExecutionID: f.executionID, ExpectedState: runstore.StageRunning, TerminationSchemaVersion: contracts.APIVersion,
				Termination: runstore.StageTermination{Outcome: runstore.TerminationInterrupted, Code: "test_abort", Message: "test abort", Phase: runstore.TerminationRunning, OccurredAt: time.Now()},
				AbortID:     "abort-fenced", Deadline: time.Now().Add(time.Minute), Reason: runstore.Reason{Code: "stage_interrupted"},
			})
		},
		"result progression":      func(c context.Context) error { return f.persistence.CommitResultProgression(c, result) },
		"termination progression": func(c context.Context) error { return f.persistence.CommitTerminationProgression(c, term) },
		"invalid state": func(c context.Context) error {
			return f.persistence.FailRunWithActiveStages(c, "run-1", state, runstore.RunFailed, runstore.Reason{Code: "scheduler_state_invalid"}, runstore.StageTermination{}, "abort-review")
		},
	}
	for name, call := range calls {
		t.Run(name, func(t *testing.T) {
			for label, c := range map[string]context.Context{"replaced": old, "missing": ctx, "wrong Run": WithRunClaim(ctx, "other-run", "competing-lane")} {
				if err := call(c); !errors.Is(err, ErrClaimLost) {
					t.Errorf("%s scope = %v, want claim loss", label, err)
				}
			}
		})
	}
	// Cancellation uses the same exact claim even after ordinary execution
	// cancellation; the cleanup context preserves the original scoped identity.
	if _, err := f.store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}
	cancelled, cancelExecution := context.WithCancel(old)
	cancelExecution()
	cleanup, cancelCleanup := claimCleanupContext(cancelled)
	defer cancelCleanup()
	for name, call := range map[string]func() error{
		"accept candidate": func() error {
			return f.persistence.AcceptResultDuringCancellation(cleanup, "run-1", f.executionID, f.result)
		},
		"finish cancellation": func() error { return f.persistence.FinishCancelledRun(cleanup, "run-1") },
		"finish termination": func() error {
			return f.persistence.CommitTerminationAndFinishRun(cleanup, "run-1", f.executionID, runstore.RunCancelling, runstore.RunCancelled, runstore.Reason{Code: runstore.CancellationUserRequested})
		},
	} {
		if err := call(); !errors.Is(err, ErrClaimLost) {
			t.Errorf("%s = %v, want claim loss", name, err)
		}
	}
	run, err := f.store.GetRun(ctx, "run-1")
	if err != nil || run.SchedulerClaim == nil || run.SchedulerClaim.ClaimID != "competing-lane" || run.State != runstore.RunCancelling {
		t.Fatalf("fenced writes changed Run: state=%s claim=%+v err=%v", run.State, run.SchedulerClaim, err)
	}
}

func TestPostgresUnadmittedWritesRejectReplacedClaim(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	h := newDeferredPlacementHarness(t, ctx, nil)
	if _, err := h.pool.Exec(ctx, `UPDATE workflow_runs SET state='pending',started_at=NULL,state_reason_code='awaiting_admission' WHERE run_id='run-1'`); err != nil {
		t.Fatal(err)
	}
	old := fixtureClaim(t, ctx, h.pool, "run-1")
	run, err := h.store.GetRun(ctx, "run-1")
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := decodeExecutableWorkflow(run)
	if err != nil {
		t.Fatal(err)
	}
	stage, err := h.scheduler.createStageExecution(old, run, workflow)
	if err != nil {
		t.Fatal(err)
	}
	requirements, err := auditBindingRequirements(run, workflow, stage.StageContext)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.workers.allocator.ReserveAllContext(ctx, stagePlacementRequest(run, workflow, stage, requirements)); err != nil {
		t.Fatal(err)
	}
	next, err := h.scheduler.buildStageCreation(ctx, run, workflow, 1, nil, stageExecutionConfiguration{variant: runstore.StageExecutionConfigBase})
	if err != nil {
		t.Fatal(err)
	}
	stealRunClaim(t, ctx, h.pool, "run-1")
	if _, err := h.scheduler.persistence.CreateStageWithContext(old, next.Params, next.ContextPins); !errors.Is(err, ErrClaimLost) {
		t.Fatalf("stale creation=%v", err)
	}
	if err := h.scheduler.persistence.DiscardUnadmittedPlacement(old, "run-1", stage.StageExecutionID, func() error { return h.workers.allocator.DiscardUnprepared(stage.StageExecutionID) }); !errors.Is(err, ErrClaimLost) {
		t.Fatalf("stale discard=%v", err)
	}
	rows, err := h.store.ListStageAllocations(ctx, stage.StageExecutionID)
	if err != nil || len(rows) != 1 || len(h.workers.allocator.grants) != 1 {
		t.Fatalf("stale discard changed placement: rows=%d grants=%d err=%v", len(rows), len(h.workers.allocator.grants), err)
	}
}
