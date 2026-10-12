package scheduler

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// executeRun performs one progression attempt of the claimed run. It reads
// the Run's state and its active StageExecution, not its Stage history
// (SCH-R164).
func (s *Scheduler) executeRun(ctx context.Context, run runstore.WorkflowRun, claimID string) error {
	// The claim returned the whole Run, and what a progression attempt uses
	// of it changes only with its state (cancellation, a model-route wait;
	// Skill initialization reads its own copy), so only a changed state
	// makes the attempt read the whole Run again.
	state, err := s.store.GetRunState(ctx, run.RunID)
	if err != nil {
		return err
	}
	if state != run.State {
		if run, err = s.store.GetRun(ctx, run.RunID); err != nil {
			return err
		}
	}
	if run.State == runstore.RunInitializing &&
		run.StateReason.Code == runstore.SkillInitializationPendingReason {
		if s.options.RunSkills == nil {
			return fmt.Errorf("Run Skill initializer is not configured")
		}
		run, err = s.options.RunSkills.InitializeRunSkills(ctx, run.RunID)
		if err != nil {
			return err
		}
	}
	if run.State == runstore.RunCancelling {
		return s.executeCancelling(ctx, run, claimID)
	}
	if run.State != runstore.RunRunning && run.State != runstore.RunPending && run.State != runstore.RunWaiting {
		return nil
	}
	workflow, err := s.workflows.executableWorkflow(run)
	if err != nil {
		return s.failUnsupportedRun(ctx, run.RunID, err)
	}
	execution, err := s.store.GetActiveStageExecution(ctx, run.RunID)
	switch {
	case err == nil:
		workflow, err = workflow.selectStage(execution.StageName)
		if err != nil {
			return s.failInvalidRunState(ctx, run.RunID, err)
		}
	case errors.Is(err, runstore.ErrNotFound):
		// Only a Run whose first Stage was never created has no active
		// StageExecution: every later Stage, retry and manual continuation is
		// created in the transaction that ends its predecessor.
		executions, listErr := s.store.ListStageExecutions(ctx, run.RunID)
		if listErr != nil {
			return listErr
		}
		if len(executions) != 0 {
			return s.failInvalidRunState(
				ctx, run.RunID, fmt.Errorf("running WorkflowRun has no active StageExecution"),
			)
		}
		execution, err = s.createStageExecution(ctx, run, workflow)
		if err != nil {
			return err
		}
	case errors.Is(err, runstore.ErrMultipleActiveStages):
		return s.failInvalidRunState(ctx, run.RunID, err)
	default:
		return err
	}
	workflow, err = workflow.selectExecution(execution)
	if err != nil {
		return s.failInvalidRunState(ctx, run.RunID, err)
	}
	if err := validatePersistedExecution(execution, run, workflow); err != nil {
		return s.failInvalidRunState(ctx, run.RunID, err)
	}
	// Scope allocation-loss interrupts to the Stage this lane is now
	// progressing; a loss for an already-finished Stage must not cancel it.
	s.setActiveStage(run.RunID, claimID, execution.StageExecutionID)

	switch execution.State {
	case runstore.StagePreparing:
		if missing := missingRequiredContext(execution); missing != "" {
			return s.beginAbort(ctx, run, workflow, execution, nil, planner.Failure{
				Code:      "context_artifact_missing",
				Message:   "Required Stage context artifact is unavailable",
				Retryable: false,
			})
		}
		return s.prepareAndPlan(ctx, run, workflow, execution)
	case runstore.StageRunning:
		return s.prepareAndPlan(ctx, run, workflow, execution)
	case runstore.StageFinalizing:
		return s.resumeFinalizing(ctx, run, workflow, execution, nil)
	case runstore.StageAborting:
		return s.resumeAborting(ctx, run, workflow, execution, nil)
	case runstore.StageInterrupted, runstore.StageCancelled, runstore.StageSucceeded, runstore.StageFailed:
		return s.failInvalidRunState(ctx, run.RunID, fmt.Errorf("running WorkflowRun selected a terminal StageExecution"))
	default:
		return s.failInvalidRunState(ctx, run.RunID, fmt.Errorf("unknown StageExecution state %q", execution.State))
	}
}

func (s *Scheduler) executeCancelling(ctx context.Context, run runstore.WorkflowRun, claimID string) error {
	workflow, workflowErr := s.workflows.executableWorkflow(run)
	if workflowErr != nil {
		s.options.Logger.Warn("cancelling Run has an unreadable Workflow snapshot", "run_id", run.RunID)
	}
	execution, err := s.store.GetActiveStageExecution(ctx, run.RunID)
	if errors.Is(err, runstore.ErrNotFound) {
		// No Stage is active: the cancellation ends the Run once every
		// earlier attempt's allocations are fenced and released, which this
		// one-time path reads the Stage history for.
		executions, listErr := s.store.ListStageExecutions(ctx, run.RunID)
		if listErr != nil {
			return listErr
		}
		for _, execution := range executions {
			_ = s.fenceRecordedAllocations(ctx, execution.StageExecutionID, nil)
			if workflowErr == nil {
				reservations := s.existingLiveReservations(ctx, run, workflow, execution)
				_ = s.releaseTerminal(execution.StageExecutionID, reservations)
			}
		}
		return s.finishCancelledRun(ctx, run.RunID)
	}
	if err != nil {
		return fmt.Errorf("cancelling Run %q: %w", run.RunID, err)
	}

	s.setActiveStage(run.RunID, claimID, execution.StageExecutionID)
	var reservations []controlplane.Reservation
	if workflowErr == nil {
		workflow, workflowErr = workflow.selectExecution(execution)
	}
	if workflowErr == nil {
		reservations = s.existingLiveReservations(ctx, run, workflow, execution)
	}
	switch execution.State {
	case runstore.StagePreparing, runstore.StageRunning:
		return s.beginCancellationAbort(ctx, run, workflow, execution, reservations)
	case runstore.StageAborting:
		return s.resumeAborting(ctx, run, workflow, execution, reservations)
	case runstore.StageFinalizing:
		return s.resumeFinalizing(ctx, run, workflow, execution, reservations)
	default:
		return fmt.Errorf("unhandled cancelling StageExecution state %q", execution.State)
	}
}

func (s *Scheduler) finishCancelledRun(ctx context.Context, runID string) error {
	operationContext, cancel := s.terminalOperationContext(ctx)
	defer cancel()
	// The commit freezes the outputs that earlier Stages bound (ART-R94).
	err := s.persistence.FinishCancelledRun(operationContext, runID)
	if errors.Is(err, runstore.ErrConflict) {
		state, loadErr := s.store.GetRunState(operationContext, runID)
		if loadErr == nil && state == runstore.RunCancelled {
			return nil
		}
	}
	return err
}

func (s *Scheduler) failUnsupportedRun(ctx context.Context, runID string, cause error) error {
	return s.failActiveRun(ctx, runID, "unsupported_workflow_shape", cause)
}
func (s *Scheduler) failInvalidRunState(ctx context.Context, runID string, cause error) error {
	return s.failActiveRun(ctx, runID, "scheduler_state_invalid", cause)
}
func (s *Scheduler) failActiveRun(ctx context.Context, runID, code string, cause error) error {
	operationContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
	defer cancel()
	state, err := s.store.GetRunState(operationContext, runID)
	if err != nil {
		return errors.Join(cause, err)
	}
	next, reason := runstore.RunFailed, runstore.Reason{Code: code}
	termination := runstore.StageTermination{
		Outcome: runstore.TerminationInterrupted, Code: code,
		Message: "Scheduler could not progress the WorkflowRun", OccurredAt: s.now(),
	}
	if state == runstore.RunCancelling {
		// Cancellation already owns the outcome; cancelling may only end as
		// cancelled, so failing it would leave a poison Run.
		next, reason = runstore.RunCancelled, runstore.Reason{Code: runstore.CancellationUserRequested}
		termination.Outcome = runstore.TerminationCancelled
	}
	abortID, err := s.newID("abort_")
	if err != nil {
		return errors.Join(cause, err)
	}
	// Active Stages end with the Run so terminal recovery releases their
	// allocations; this is a single commit and never re-enters Stage recovery.
	err = s.persistence.FailRunWithActiveStages(operationContext, runID, state, next, reason, termination, abortID)
	if err != nil {
		return errors.Join(cause, err)
	}
	return nil
}

func infrastructureFailure(code, message string, cause error) planner.Failure {
	retryable := true
	var apiError *controlplane.RuntimeAPIError
	if errors.As(cause, &apiError) {
		retryable = apiError.Retryable
	}
	return planner.Failure{Code: code, Message: message, Retryable: retryable}
}
