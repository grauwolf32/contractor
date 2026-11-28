package scheduler

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// V498-002: Complete/Abort in the local TLA+ model are these claimed
// transactions. Forced orders cover both winners independently of scheduling;
// the concurrent case also exercises PostgreSQL's Run-then-Stage serialization.
func TestPostgresStagePilotCompetingTerminalPhases(t *testing.T) {
	for _, order := range []string{"result-first", "abort-first", "concurrent"} {
		t.Run(order, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			pool := isolatedSchedulerPool(t, ctx)
			f := createRunningStageFixture(t, ctx, pool)
			claimed := fixtureClaim(t, ctx, pool, "run-1")
			finalize, abort := stagePilotPhaseParams(f)
			complete := func() error { return f.persistence.EnterFinalizingWithResult(claimed, finalize) }
			interrupt := func() error { return f.persistence.EnterAbortingWithTermination(claimed, "run-1", abort) }
			var resultErr, abortErr error
			switch order {
			case "result-first":
				resultErr, abortErr = complete(), interrupt()
			case "abort-first":
				abortErr, resultErr = interrupt(), complete()
			case "concurrent":
				start := make(chan struct{})
				results, aborts := make(chan error, 1), make(chan error, 1)
				go func() { <-start; results <- complete() }()
				go func() { <-start; aborts <- interrupt() }()
				close(start)
				resultErr, abortErr = <-results, <-aborts
			}
			if (resultErr == nil) == (abortErr == nil) ||
				(resultErr != nil && !errors.Is(resultErr, runstore.ErrConflict)) ||
				(abortErr != nil && !errors.Is(abortErr, runstore.ErrConflict)) {
				t.Fatalf("expected one CAS winner: result=%v abort=%v", resultErr, abortErr)
			}
			if order == "result-first" && resultErr != nil || order == "abort-first" && abortErr != nil {
				t.Fatal("first durable writer did not win")
			}
			saved, err := f.store.GetStageExecution(ctx, f.executionID)
			if err != nil {
				t.Fatal(err)
			}
			if resultErr == nil {
				if saved.State != runstore.StageFinalizing || saved.Termination != nil ||
					!reflect.DeepEqual(saved.CandidateResult, &f.result) || saved.FinalizationID == nil ||
					*saved.FinalizationID != finalize.FinalizationID || !saved.FinalizationDeadline.Equal(finalize.Deadline) {
					t.Fatalf("finalization winner was replaced: %+v", saved)
				}
				changed := stagePilotProgression(f)
				changed.Result.Summary = "late different result"
				if err := f.persistence.CommitResultProgression(claimed, changed); !errors.Is(err, runstore.ErrConflict) {
					t.Fatalf("changed candidate acceptance = %v", err)
				}
				after, err := f.store.GetStageExecution(ctx, f.executionID)
				if err != nil || !reflect.DeepEqual(saved, after) {
					t.Fatalf("rejected candidate mutated Stage: %v", err)
				}
				if err := f.persistence.CommitResultProgression(claimed, stagePilotProgression(f)); err != nil {
					t.Fatal(err)
				}
				stagePilotCheckEffects(t, ctx, f, runstore.RunSucceeded, 1, 1)
			} else {
				if saved.State != runstore.StageAborting || saved.CandidateResult != nil ||
					!reflect.DeepEqual(saved.Termination, &abort.Termination) || saved.AbortID == nil ||
					*saved.AbortID != abort.AbortID || !saved.AbortDeadline.Equal(abort.Deadline) {
					t.Fatalf("abort winner was replaced: %+v", saved)
				}
				stagePilotCancel(t, ctx, f)
				if err := f.persistence.CommitTerminationAndFinishRun(claimed, "run-1", f.executionID,
					runstore.RunCancelling, runstore.RunCancelled, runstore.Reason{Code: runstore.CancellationUserRequested}); err != nil {
					t.Fatal(err)
				}
				finished, err := f.store.GetStageExecution(ctx, f.executionID)
				if err != nil || finished.State != runstore.StageInterrupted || !reflect.DeepEqual(saved.Termination, finished.Termination) {
					t.Fatalf("cancellation replaced the recorded interruption: %+v, %v", finished, err)
				}
				stagePilotCheckEffects(t, ctx, f, runstore.RunCancelled, 0, 0)
			}
			terminal, err := f.store.GetStageExecution(ctx, f.executionID)
			if err != nil {
				t.Fatal(err)
			}
			// Late phase writes and replayed publication cannot modify the terminal fact.
			for _, err := range []error{complete(), interrupt(), f.persistence.CommitResultProgression(claimed, stagePilotProgression(f))} {
				if !errors.Is(err, runstore.ErrConflict) {
					t.Fatalf("terminal replay = %v", err)
				}
			}
			after, err := f.store.GetStageExecution(ctx, f.executionID)
			if err != nil || !reflect.DeepEqual(terminal, after) {
				t.Fatalf("terminal fact changed: %v", err)
			}
		})
	}
}

func TestPostgresStagePilotCancellationOrder(t *testing.T) {
	for _, order := range []string{"cancel-before-candidate", "candidate-then-cancel", "commit-then-cancel"} {
		t.Run(order, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			pool := isolatedSchedulerPool(t, ctx)
			f := createRunningStageFixture(t, ctx, pool)
			claimed := fixtureClaim(t, ctx, pool, "run-1")
			finalize, abort := stagePilotPhaseParams(f)
			if order == "cancel-before-candidate" {
				stagePilotCancel(t, ctx, f)
				if err := f.persistence.EnterFinalizingWithResult(claimed, finalize); !errors.Is(err, runstore.ErrConflict) {
					t.Fatalf("late candidate = %v", err)
				}
				abort.Termination.Outcome = runstore.TerminationCancelled
				if err := f.persistence.EnterAbortingWithTermination(claimed, "run-1", abort); err != nil {
					t.Fatal(err)
				}
				if err := f.persistence.CommitTerminationAndFinishRun(claimed, "run-1", f.executionID,
					runstore.RunCancelling, runstore.RunCancelled, runstore.Reason{Code: runstore.CancellationUserRequested}); err != nil {
					t.Fatal(err)
				}
			} else {
				if err := f.persistence.EnterFinalizingWithResult(claimed, finalize); err != nil {
					t.Fatal(err)
				}
				if order == "commit-then-cancel" {
					if err := f.persistence.CommitResultProgression(claimed, stagePilotProgression(f)); err != nil {
						t.Fatal(err)
					}
					stagePilotCancel(t, ctx, f)
				} else {
					stagePilotCancel(t, ctx, f)
					if err := f.persistence.EnterAbortingWithTermination(claimed, "run-1", abort); !errors.Is(err, runstore.ErrConflict) {
						t.Fatalf("cancel replaced candidate: %v", err)
					}
					if err := f.persistence.CommitResultProgression(claimed, stagePilotProgression(f)); !errors.Is(err, runstore.ErrConflict) {
						t.Fatalf("publication after cancel: %v", err)
					}
					changed := f.result.Clone()
					changed.Summary = "changed audit-only candidate"
					if err := f.persistence.AcceptResultDuringCancellation(claimed, "run-1", f.executionID, changed); !errors.Is(err, runstore.ErrConflict) {
						t.Fatalf("changed audit candidate: %v", err)
					}
					if err := f.persistence.AcceptResultDuringCancellation(claimed, "run-1", f.executionID, f.result); err != nil {
						t.Fatal(err)
					}
				}
			}
			stage, err := f.store.GetStageExecution(ctx, f.executionID)
			if err != nil {
				t.Fatal(err)
			}
			if order == "cancel-before-candidate" {
				if stage.State != runstore.StageCancelled || stage.AcceptedResult != nil || stage.Termination == nil {
					t.Fatalf("cancellation fact: %+v", stage)
				}
			} else if stage.State != runstore.StageSucceeded || stage.Termination != nil || !reflect.DeepEqual(stage.AcceptedResult, &f.result) {
				t.Fatalf("saved candidate was not accepted exactly: %+v", stage)
			}
			if order == "commit-then-cancel" {
				stagePilotCheckEffects(t, ctx, f, runstore.RunSucceeded, 1, 1)
			} else {
				stagePilotCheckEffects(t, ctx, f, runstore.RunCancelled, 0, 0)
			}
		})
	}
}

// A lost acknowledgement leaves the real durable phase in place. A new
// Scheduler must recover it without asking a Planner to produce another result.
func TestPostgresStagePilotFreshSchedulerRecovery(t *testing.T) {
	for _, phase := range []string{"finalizing", "aborting"} {
		for _, recovery := range []string{"before-deadline", "expired", "cancelled"} {
			t.Run(phase+"/"+recovery, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
				defer cancel()
				lost := &stagePilotLostAcknowledgement{}
				h := newDeferredPlacementHarness(t, ctx, func(p AtomicPersistence) AtomicPersistence { lost.AtomicPersistence = p; return lost })
				if phase == "aborting" {
					h.invoker.before = func(context.Context, contracts.WorkerHandle) error { return errors.New("injected Worker interruption") }
				}
				if worked, err := h.scheduler.RunOnce(ctx); !worked || !errors.Is(err, errStagePilotLostAcknowledgement) {
					t.Fatalf("persist phase: %v %v", worked, err)
				}
				executions, err := h.store.ListStageExecutions(ctx, "run-1")
				if err != nil || len(executions) != 1 || string(executions[0].State) != phase {
					t.Fatalf("saved phase: %+v %v", executions, err)
				}
				saved := executions[0]
				id, deadline := saved.FinalizationID, saved.FinalizationDeadline
				wantRun, wantStage := runstore.RunSucceeded, runstore.StageSucceeded
				if phase == "aborting" {
					id, deadline = saved.AbortID, saved.AbortDeadline
					wantRun, wantStage = runstore.RunFailed, runstore.StageInterrupted
				}
				if id == nil || deadline == nil || h.invoker.calls != 1 {
					t.Fatal("missing durable phase identity or initial invocation")
				}
				if recovery == "cancelled" {
					_, err = h.store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: h.scheduler.now()})
					if err != nil {
						t.Fatal(err)
					}
					wantRun = runstore.RunCancelled
				}
				options := h.scheduler.options
				options.FinalizationTimeout, options.AbortTimeout = time.Hour, time.Hour
				recoveryTime := deadline.Add(-time.Second)
				if recovery == "expired" {
					recoveryTime = deadline.Add(time.Second)
				}
				options.Clock = staticClock{now: recoveryTime}
				workers := &terminalDeadlineWorkers{memoryWorkers: h.workers}
				planners := &stagePilotNoReplayPlanners{}
				restarted, err := New(h.store, lost.AtomicPersistence, h.scheduler.artifacts, h.scheduler.allocator, workers, planners, options)
				if err != nil {
					t.Fatal(err)
				}
				if worked, err := restarted.RunOnce(ctx); !worked || err != nil {
					t.Fatalf("recover: %v %v", worked, err)
				}
				if recovery == "expired" {
					if workers.operationID != "" {
						t.Fatal("expired drain budget was renewed")
					}
				} else if workers.operationID != *id || !workers.deadline.Equal(*deadline) {
					t.Fatalf("recovery changed drain identity/deadline: %q %v", workers.operationID, workers.deadline)
				}
				finished, err := h.store.GetStageExecution(ctx, saved.StageExecutionID)
				if err != nil || finished.State != wantStage || !reflect.DeepEqual(saved.CandidateResult, finished.CandidateResult) || !reflect.DeepEqual(saved.Termination, finished.Termination) ||
					!reflect.DeepEqual(saved.FinalizationID, finished.FinalizationID) || !reflect.DeepEqual(saved.FinalizationDeadline, finished.FinalizationDeadline) ||
					!reflect.DeepEqual(saved.AbortID, finished.AbortID) || !reflect.DeepEqual(saved.AbortDeadline, finished.AbortDeadline) {
					t.Fatalf("recovered durable fact: %+v %v", finished, err)
				}
				run, err := h.store.GetRun(ctx, "run-1")
				if err != nil || run.State != wantRun {
					t.Fatalf("recovered Run: %s %v", run.State, err)
				}
				if worked, err := restarted.RunOnce(ctx); worked || err != nil {
					t.Fatalf("terminal recovery replay: %v %v", worked, err)
				}
				if planners.calls != 0 || h.invoker.calls != 1 {
					t.Fatalf("repeated semantic work: planner=%d Worker=%d", planners.calls, h.invoker.calls)
				}
			})
		}
	}
}

func stagePilotPhaseParams(f finalizingFixture) (runstore.EnterFinalizingParams, runstore.EnterAbortingParams) {
	now := time.Date(2026, 11, 28, 12, 0, 0, 0, time.UTC)
	return runstore.EnterFinalizingParams{StageExecutionID: f.executionID, ResultSchemaVersion: contracts.APIVersion, Candidate: f.result,
			FinalizationID: "pilot-finalize", Deadline: now.Add(time.Minute), Reason: runstore.Reason{Code: "planner_completed"}},
		runstore.EnterAbortingParams{StageExecutionID: f.executionID, ExpectedState: runstore.StageRunning, TerminationSchemaVersion: contracts.APIVersion,
			Termination: runstore.StageTermination{Outcome: runstore.TerminationInterrupted, Code: "pilot_interrupted", Message: "interrupted", Phase: runstore.TerminationRunning, OccurredAt: now},
			AbortID:     "pilot-abort", Deadline: now.Add(2 * time.Minute), Reason: runstore.Reason{Code: "stage_interrupted"}}
}

func TestPostgresStagePilotRunningPlannerRecovery(t *testing.T) {
	for _, cancelled := range []bool{false, true} {
		name := "interrupted"
		if cancelled {
			name = "cancelled"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			stop := &stagePilotStopBeforeAbort{}
			h := newDeferredPlacementHarness(t, ctx, func(p AtomicPersistence) AtomicPersistence { stop.AtomicPersistence = p; return stop })
			h.invoker.before = func(context.Context, contracts.WorkerHandle) error { return errors.New("injected process loss") }
			if worked, err := h.scheduler.RunOnce(ctx); !worked || !errors.Is(err, errStagePilotLostAcknowledgement) {
				t.Fatalf("leave running: %v %v", worked, err)
			}
			stages, err := h.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(stages) != 1 || stages[0].State != runstore.StageRunning || stages[0].CandidateResult != nil || stages[0].Termination != nil {
				t.Fatalf("running durable state: %+v %v", stages, err)
			}
			wantRun, wantStage, wantCode := runstore.RunFailed, runstore.StageInterrupted, "control_plane_state_lost"
			if cancelled {
				if _, err := h.store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: h.scheduler.now()}); err != nil {
					t.Fatal(err)
				}
				wantRun, wantStage, wantCode = runstore.RunCancelled, runstore.StageCancelled, runstore.CancellationUserRequested
			}
			planners := &stagePilotNoReplayPlanners{}
			// A fresh Control Plane has no live allocation grants. Keep the real
			// durable Planner session so recovery must inspect its saved state.
			restarted, err := New(h.store, stop.AtomicPersistence, h.scheduler.artifacts, laneNoopAllocator{}, laneNoopWorkers{}, planners, h.scheduler.options)
			if err != nil {
				t.Fatal(err)
			}
			if worked, err := restarted.RunOnce(ctx); !worked || err != nil {
				t.Fatalf("recover running: %v %v", worked, err)
			}
			stage, err := h.store.GetStageExecution(ctx, stages[0].StageExecutionID)
			if err != nil || stage.State != wantStage || stage.Termination == nil || stage.Termination.Code != wantCode || stage.AcceptedResult != nil {
				t.Fatalf("running recovery fact: %+v %v", stage, err)
			}
			run, err := h.store.GetRun(ctx, "run-1")
			if err != nil || run.State != wantRun {
				t.Fatalf("running recovery Run: %s %v", run.State, err)
			}
			if planners.calls != 0 || h.invoker.calls != 1 {
				t.Fatal("recovery repeated Planner/Worker work")
			}
		})
	}
}

type stagePilotStopBeforeAbort struct{ AtomicPersistence }

func (*stagePilotStopBeforeAbort) EnterAbortingWithTermination(context.Context, string, runstore.EnterAbortingParams) error {
	return errStagePilotLostAcknowledgement
}

func stagePilotProgression(f finalizingFixture) ResultProgression {
	return ResultProgression{RunID: "run-1", StageExecutionID: f.executionID, Result: f.result.Clone(), WorkflowOutputs: f.workflow.Stages["copy"].WorkflowOutputs,
		OutputContracts: f.workflow.Outputs, Progression: terminalSuccessProgression("run-1", f.executionID)}
}

func stagePilotCancel(t *testing.T, ctx context.Context, f finalizingFixture) {
	t.Helper()
	if _, err := f.store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested,
		RequestedAt: time.Date(2026, 11, 28, 12, 0, 0, 0, time.UTC)}); err != nil {
		t.Fatal(err)
	}
}

func stagePilotCheckEffects(t *testing.T, ctx context.Context, f finalizingFixture, state runstore.WorkflowRunState, outputs, decisions int) {
	t.Helper()
	run, err := f.store.GetRun(ctx, "run-1")
	if err != nil || run.State != state {
		t.Fatalf("Run: %s %v", run.State, err)
	}
	scoped, err := f.artifacts.Run("run-1")
	if err != nil {
		t.Fatal(err)
	}
	namespace := "outputs"
	page, err := scoped.ListRefs(ctx, artifacts.RefPageQuery{Namespace: &namespace, Limit: 10})
	if err != nil || len(page.Refs) != outputs {
		t.Fatalf("outputs: %+v %v", page, err)
	}
	transitions, err := f.store.ListStageTransitionDecisions(ctx, "run-1")
	if err != nil || len(transitions) != decisions {
		t.Fatalf("decisions: %+v %v", transitions, err)
	}
}

var errStagePilotLostAcknowledgement = errors.New("injected lost phase-commit acknowledgement")

type stagePilotLostAcknowledgement struct{ AtomicPersistence }

func (p *stagePilotLostAcknowledgement) EnterFinalizingWithResult(ctx context.Context, params runstore.EnterFinalizingParams) error {
	if err := p.AtomicPersistence.EnterFinalizingWithResult(ctx, params); err != nil {
		return err
	}
	return errStagePilotLostAcknowledgement
}

func (p *stagePilotLostAcknowledgement) EnterAbortingWithTermination(ctx context.Context, runID string, params runstore.EnterAbortingParams) error {
	if err := p.AtomicPersistence.EnterAbortingWithTermination(ctx, runID, params); err != nil {
		return err
	}
	return errStagePilotLostAcknowledgement
}

type stagePilotNoReplayPlanners struct{ calls int }

func (p *stagePilotNoReplayPlanners) Create(string, planner.Invocation) (planner.Planner, error) {
	p.calls++
	return nil, errors.New("unexpected Planner invocation during durable-phase recovery")
}
