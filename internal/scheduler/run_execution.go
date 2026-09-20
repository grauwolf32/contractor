package scheduler

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (s *Scheduler) executeRun(ctx context.Context, run runstore.WorkflowRun) error {
	current, err := s.store.GetRun(ctx, run.RunID)
	if err != nil {
		return err
	}
	run = current
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
		return s.executeCancelling(ctx, run)
	}
	if run.State != runstore.RunRunning && run.State != runstore.RunPending && run.State != runstore.RunWaiting {
		return nil
	}
	workflow, err := decodeExecutableWorkflow(run)
	if err != nil {
		return s.failUnsupportedRun(ctx, run.RunID, err)
	}
	executions, err := s.store.ListStageExecutions(ctx, run.RunID)
	if err != nil {
		return err
	}
	var execution runstore.StageExecution
	if len(executions) == 0 {
		execution, err = s.createStageExecution(ctx, run, workflow)
		if err != nil {
			return err
		}
	} else {
		active := make([]runstore.StageExecution, 0, 1)
		for _, candidate := range executions {
			switch candidate.State {
			case runstore.StagePreparing, runstore.StageRunning, runstore.StageFinalizing, runstore.StageAborting:
				active = append(active, candidate)
			}
		}
		if len(active) != 1 {
			return s.failInvalidRunState(
				ctx,
				run.RunID,
				fmt.Errorf("running WorkflowRun has %d active StageExecutions", len(active)),
			)
		}
		execution = active[0]
		workflow, err = workflow.selectStage(execution.StageName)
		if err != nil {
			return s.failInvalidRunState(ctx, run.RunID, err)
		}
	}
	workflow, err = workflow.selectExecution(execution)
	if err != nil {
		return s.failInvalidRunState(ctx, run.RunID, err)
	}
	if err := validatePersistedExecution(execution, run, workflow); err != nil {
		return s.failInvalidRunState(ctx, run.RunID, err)
	}

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

func (s *Scheduler) executeCancelling(ctx context.Context, run runstore.WorkflowRun) error {
	executions, err := s.store.ListStageExecutions(ctx, run.RunID)
	if err != nil {
		return err
	}
	workflow, workflowErr := decodeExecutableWorkflow(run)
	if workflowErr != nil {
		s.options.Logger.Warn("cancelling Run has an unreadable Workflow snapshot", "run_id", run.RunID)
	}

	active := make([]runstore.StageExecution, 0, 1)
	for _, execution := range executions {
		switch execution.State {
		case runstore.StagePreparing, runstore.StageRunning, runstore.StageFinalizing, runstore.StageAborting:
			active = append(active, execution)
		}
	}
	if len(active) > 1 {
		return fmt.Errorf("cancelling MVP Run %q has multiple active StageExecutions", run.RunID)
	}
	if len(active) == 0 {
		for _, execution := range executions {
			_ = s.fenceRecordedAllocations(ctx, execution.StageExecutionID, nil)
			if workflowErr == nil {
				reservations := s.existingLiveReservations(ctx, run, workflow, execution)
				_ = s.releaseTerminal(execution.StageExecutionID, reservations)
			}
		}
		return s.finishCancelledRun(ctx, run.RunID)
	}

	execution := active[0]
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
	_, err := s.store.TransitionRun(
		operationContext,
		runID,
		runstore.RunCancelling,
		runstore.RunCancelled,
		runstore.Reason{Code: runstore.CancellationUserRequested},
	)
	if errors.Is(err, runstore.ErrConflict) {
		current, loadErr := s.store.GetRun(operationContext, runID)
		if loadErr == nil && current.State == runstore.RunCancelled {
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
	current, err := s.store.GetRun(operationContext, runID)
	if err != nil {
		return errors.Join(cause, err)
	}
	_, err = s.store.TransitionRun(operationContext, runID, current.State, runstore.RunFailed, runstore.Reason{Code: code})
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
