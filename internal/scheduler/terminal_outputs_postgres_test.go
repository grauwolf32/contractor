package scheduler

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// When a Run becomes failed or cancelled, the outputs its earlier Stages bound
// are frozen in the same transaction (ART-R94, LIFE-ART-2): when the Scheduler
// ends a Run it cannot progress, whether the Run fails or cancellation owns it
// (SCH-R102), and when cancellation ends a Run without an active Stage.
func TestPostgresFailedOrCancelledRunFreezesEarlierOutputs(t *testing.T) {
	for _, test := range []struct {
		name        string
		end         func(*testing.T, context.Context, deferredPlacementHarness)
		runState    runstore.WorkflowRunState
		reviewState runstore.StageExecutionState
	}{
		{"unprogressable Run fails", func(t *testing.T, ctx context.Context, h deferredPlacementHarness) {
			if err := h.scheduler.failInvalidRunState(ctx, "run-1", errReviewStateInvalid); err != nil {
				t.Fatal(err)
			}
		}, runstore.RunFailed, runstore.StageInterrupted},
		{"unprogressable cancelling Run is cancelled", func(t *testing.T, ctx context.Context, h deferredPlacementHarness) {
			requestTestCancellation(t, ctx, h.store)
			if err := h.scheduler.failInvalidRunState(ctx, "run-1", errReviewStateInvalid); err != nil {
				t.Fatal(err)
			}
		}, runstore.RunCancelled, runstore.StageCancelled},
		{"cancelling Run without an active Stage", func(t *testing.T, ctx context.Context, h deferredPlacementHarness) {
			requestTestCancellation(t, ctx, h.store)
			// Every valid path ends the last Stage together with its Run, so
			// a cancelling Run without an active Stage normally never started
			// one. Ending the review Stage alone gives the commit that ends
			// such a Run outputs to freeze.
			review, err := h.store.GetActiveStageExecution(ctx, "run-1")
			if err != nil {
				t.Fatal(err)
			}
			if err := h.scheduler.persistence.EnterAbortingWithTermination(ctx, "run-1", runstore.EnterAbortingParams{
				StageExecutionID: review.StageExecutionID, ExpectedState: runstore.StagePreparing,
				TerminationSchemaVersion: contracts.APIVersion,
				Termination: runstore.StageTermination{
					Outcome: runstore.TerminationCancelled, Code: runstore.CancellationUserRequested,
					Message: "WorkflowRun cancellation was requested", Phase: runstore.TerminationPreparing,
					OccurredAt: time.Now().UTC(),
				},
				AbortID: "abort-review", Deadline: time.Now().Add(time.Minute),
				Reason: runstore.Reason{Code: "stage_cancelled"},
			}); err != nil {
				t.Fatal(err)
			}
			if err := h.store.CompleteStageTermination(ctx, review.StageExecutionID); err != nil {
				t.Fatal(err)
			}
			if worked, err := h.scheduler.RunOnce(ctx); err != nil || !worked {
				t.Fatalf("cancellation RunOnce = (%v, %v)", worked, err)
			}
		}, runstore.RunCancelled, runstore.StageCancelled},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			h, runArtifacts, output := newEarlierOutputHarness(t, ctx)

			test.end(t, ctx, h)

			run, err := h.store.GetRun(ctx, "run-1")
			if err != nil || run.State != test.runState {
				t.Fatalf("Run = (%s %+v, %v), want %s", run.State, run.StateReason, err, test.runState)
			}
			executions, err := h.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(executions) != 2 || executions[0].State != runstore.StageSucceeded ||
				executions[1].State != test.reviewState {
				t.Fatalf("StageExecutions = (%v, %v), want build succeeded and review %s",
					stageExecutionStates(executions), err, test.reviewState)
			}
			requireFrozenOutput(t, ctx, runArtifacts, output)
		})
	}
}

// errReviewStateInvalid is the cause these tests give for a Run the
// Scheduler cannot progress.
var errReviewStateInvalid = errors.New("review Stage state is invalid")

// newEarlierOutputHarness runs the first Stage of a two-Stage Workflow: build
// maps its result to outputs/result and hands over to review, which is
// preparing. It returns the binding build created, still unfrozen while the
// Run is running.
func newEarlierOutputHarness(
	t *testing.T, ctx context.Context,
) (deferredPlacementHarness, artifacts.ScopedStore, contracts.ArtifactRef) {
	t.Helper()
	h := newDeferredPlacementHarness(t, ctx, nil, func(workflow *resolved.ResolvedWorkflow) {
		stage := workflow.Stages[workflow.EntryStage]
		build, review := stage, stage
		build.On.Succeeded = resolved.TransitionAction{Kind: resolved.TransitionNext, NextStage: "review"}
		review.WorkflowOutputs = map[string]string{}
		workflow.EntryStage = "build"
		workflow.Stages = map[string]resolved.ResolvedStage{"build": build, "review": review}
	})
	if err := resolved.ValidateWorkflowGraph(h.workflow); err != nil {
		t.Fatal(err)
	}
	if worked, err := h.scheduler.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("build RunOnce = (%v, %v)", worked, err)
	}
	executions, err := h.store.ListStageExecutions(ctx, "run-1")
	if err != nil || len(executions) != 2 || executions[0].State != runstore.StageSucceeded ||
		executions[1].State != runstore.StagePreparing {
		t.Fatalf("StageExecutions after build = (%v, %v), want build succeeded and review preparing",
			stageExecutionStates(executions), err)
	}
	runArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(h.pool)).Run("run-1")
	if err != nil {
		t.Fatal(err)
	}
	output, err := runArtifacts.Metadata(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "result"})
	if err != nil || output.Frozen {
		t.Fatalf("output of build = (%+v, %v), want bound and not yet frozen", output, err)
	}
	return h, runArtifacts, output.Ref
}

// Every commit that makes a Run failed or cancelled freezes the outputs the
// Run already bound (ART-R94): Workflow failure after a result or a
// termination, acceptance and termination during cancellation, the end of a
// Run the Scheduler cannot progress (SCH-R102) and the end of a cancelling
// Run without an active Stage. Each case binds outputs/result, as an earlier
// Stage would have, before the commit.
func TestPostgresEveryFailedOrCancelledCommitFreezesOutputs(t *testing.T) {
	interrupted := func(phase runstore.TerminationPhase) runstore.StageTermination {
		return runstore.StageTermination{
			Outcome: runstore.TerminationInterrupted, Code: "scheduler_state_invalid",
			Message: "Scheduler could not progress the WorkflowRun", Phase: phase, OccurredAt: time.Now().UTC(),
		}
	}
	cancelled := func(phase runstore.TerminationPhase) runstore.StageTermination {
		return runstore.StageTermination{
			Outcome: runstore.TerminationCancelled, Code: runstore.CancellationUserRequested,
			Message: "WorkflowRun cancellation was requested", Phase: phase, OccurredAt: time.Now().UTC(),
		}
	}
	enterAborting := func(ctx context.Context, f finalizingFixture, termination runstore.StageTermination) error {
		return f.persistence.EnterAbortingWithTermination(ctx, "run-1", runstore.EnterAbortingParams{
			StageExecutionID: f.executionID, ExpectedState: runstore.StageRunning,
			TerminationSchemaVersion: contracts.APIVersion, Termination: termination,
			AbortID: "abort-1", Deadline: time.Now().Add(time.Minute),
			Reason: runstore.Reason{Code: "stage_" + string(termination.Outcome)},
		})
	}
	for _, test := range []struct {
		name     string
		stage    runstore.StageExecutionState
		cancel   bool
		commit   func(context.Context, finalizingFixture) error
		runState runstore.WorkflowRunState
	}{
		{"Workflow failure after a result", runstore.StageFinalizing, false,
			func(ctx context.Context, f finalizingFixture) error {
				return f.persistence.CommitResultProgression(ctx, ResultProgression{
					RunID: "run-1", StageExecutionID: f.executionID, Result: f.result,
					Progression: terminalFailProgression("run-1", f.executionID),
				})
			}, runstore.RunFailed},
		{"Workflow failure after a termination", runstore.StageRunning, false,
			func(ctx context.Context, f finalizingFixture) error {
				if err := enterAborting(ctx, f, interrupted(runstore.TerminationRunning)); err != nil {
					return err
				}
				return f.persistence.CommitTerminationProgression(ctx, TerminationProgression{
					RunID: "run-1", StageExecutionID: f.executionID,
					Progression: terminalFailProgression("run-1", f.executionID),
				})
			}, runstore.RunFailed},
		{"result accepted during cancellation", runstore.StageFinalizing, true,
			func(ctx context.Context, f finalizingFixture) error {
				return f.persistence.AcceptResultDuringCancellation(ctx, "run-1", f.executionID, f.result)
			}, runstore.RunCancelled},
		{"termination committed during cancellation", runstore.StageRunning, true,
			func(ctx context.Context, f finalizingFixture) error {
				if err := enterAborting(ctx, f, cancelled(runstore.TerminationRunning)); err != nil {
					return err
				}
				return f.persistence.CommitTerminationAndFinishRun(
					ctx, "run-1", f.executionID, runstore.RunCancelling, runstore.RunCancelled,
					runstore.Reason{Code: runstore.CancellationUserRequested},
				)
			}, runstore.RunCancelled},
		{"unprogressable Run fails", runstore.StageRunning, false,
			func(ctx context.Context, f finalizingFixture) error {
				return f.persistence.FailRunWithActiveStages(
					ctx, "run-1", runstore.RunRunning, runstore.RunFailed,
					runstore.Reason{Code: "scheduler_state_invalid"}, interrupted(runstore.TerminationPreparing), "abort-1",
				)
			}, runstore.RunFailed},
		{"unprogressable cancelling Run is cancelled", runstore.StageRunning, true,
			func(ctx context.Context, f finalizingFixture) error {
				return f.persistence.FailRunWithActiveStages(
					ctx, "run-1", runstore.RunCancelling, runstore.RunCancelled,
					runstore.Reason{Code: runstore.CancellationUserRequested}, cancelled(runstore.TerminationPreparing),
					"abort-1",
				)
			}, runstore.RunCancelled},
		{"cancelling Run without an active Stage", "", true,
			func(ctx context.Context, f finalizingFixture) error {
				s := &Scheduler{store: f.store, persistence: f.persistence, options: Options{
					OperationTimeout: 5 * time.Second, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
				}}
				return s.finishCancelledRun(ctx, "run-1")
			}, runstore.RunCancelled},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			pool := isolatedSchedulerPool(t, ctx)
			var fixture finalizingFixture
			switch test.stage {
			case runstore.StageFinalizing:
				fixture = createFinalizingFixture(t, ctx, pool)
			case runstore.StageRunning:
				fixture = createRunningStageFixture(t, ctx, pool)
			default:
				fixture.store = runstore.NewPostgresStore(pool)
				fixture.artifacts = artifacts.NewService(artifacts.NewPostgresRepository(pool))
				fixture.persistence, _ = NewPostgresPersistence(pool)
				createSchedulerRun(t, ctx, fixture.store, fixture.artifacts, loadSchedulerWorkflow(t))
			}
			runArtifacts, err := fixture.artifacts.Run("run-1")
			if err != nil {
				t.Fatal(err)
			}
			earlier, err := runArtifacts.Write(
				ctx, contracts.ArtifactRef{Namespace: "builder", Name: "earlier"},
				artifacts.Payload{MediaType: "text/plain", Data: []byte("earlier\n")}, nil,
			)
			if err != nil {
				t.Fatal(err)
			}
			bound, err := bindOutput(ctx, fixture.artifacts, "run-1", "result", earlier.Ref)
			if err != nil {
				t.Fatal(err)
			}
			if test.cancel {
				requestTestCancellation(t, ctx, fixture.store)
			}

			if err := test.commit(ctx, fixture); err != nil {
				t.Fatalf("commit: %v", err)
			}

			run, err := fixture.store.GetRun(ctx, "run-1")
			if err != nil || run.State != test.runState {
				t.Fatalf("Run = (%s %+v, %v), want %s", run.State, run.StateReason, err, test.runState)
			}
			requireFrozenOutput(t, ctx, runArtifacts, bound.TargetRef)
		})
	}
}

func terminalFailProgression(runID, sourceExecutionID string) StageProgression {
	return StageProgression{
		Decision: runstore.RecordStageTransitionDecisionParams{
			SourceExecutionID: sourceExecutionID, RunID: runID, Action: runstore.StageTransitionFail,
		},
		TerminalRunState: runstore.RunFailed,
		RunReason:        runstore.Reason{Code: "stage_failed"},
	}
}

func requestTestCancellation(t *testing.T, ctx context.Context, store *runstore.PostgresStore) {
	t.Helper()
	run, err := store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{
		Code: runstore.CancellationUserRequested, RequestedAt: time.Now(),
	})
	if err != nil || run.State != runstore.RunCancelling {
		t.Fatalf("cancellation request = (%s, %v), want cancelling", run.State, err)
	}
}

// requireFrozenOutput checks that outputs/result of run-1 is frozen at the
// exact revision want.
func requireFrozenOutput(t *testing.T, ctx context.Context, runArtifacts artifacts.ScopedStore, want contracts.ArtifactRef) {
	t.Helper()
	output, err := runArtifacts.Metadata(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "result"})
	if err != nil || !output.Frozen || !output.Ref.SameExact(want) {
		t.Fatalf("outputs/result = (%s frozen=%v, %v), want %s frozen", output.Ref.Key(), output.Frozen, err, want.Key())
	}
}
