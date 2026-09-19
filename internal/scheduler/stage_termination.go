package scheduler

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (s *Scheduler) beginAbort(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
	failure planner.Failure,
) error {
	return s.beginTermination(ctx, run, workflow, execution, reservations, runstore.StageTermination{
		Outcome: runstore.TerminationInterrupted,
		Code:    failure.Code, Message: failure.Message, Retryable: failure.Retryable,
	})
}

func (s *Scheduler) beginCancellationAbort(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
) error {
	message := "WorkflowRun cancellation was requested"
	if run.Cancellation != nil && run.Cancellation.Reason != nil {
		message = *run.Cancellation.Reason
	}
	return s.beginTermination(ctx, run, workflow, execution, reservations, runstore.StageTermination{
		Outcome: runstore.TerminationCancelled,
		Code:    runstore.CancellationUserRequested, Message: message, Retryable: false,
	})
}

func (s *Scheduler) beginTermination(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
	termination runstore.StageTermination,
) error {
	if execution.State != runstore.StagePreparing && execution.State != runstore.StageRunning {
		return fmt.Errorf("cannot abort StageExecution from %s", execution.State)
	}
	if err := s.fenceRecordedAllocations(ctx, execution.StageExecutionID, reservations); err != nil {
		return fmt.Errorf("write-fence Stage allocations: %w", err)
	}
	abortID, err := s.newID("abort_")
	if err != nil {
		return err
	}
	deadline := s.now().Add(s.options.AbortTimeout)
	phase := runstore.TerminationPreparing
	if execution.State == runstore.StageRunning {
		phase = runstore.TerminationRunning
	}
	termination.Phase = phase
	termination.OccurredAt = s.now()
	transitionContext, cancelTransition := context.WithTimeout(ctx, s.options.OperationTimeout)
	err = s.persistence.EnterAbortingWithTermination(
		transitionContext,
		run.RunID,
		runstore.EnterAbortingParams{
			StageExecutionID:         execution.StageExecutionID,
			ExpectedState:            execution.State,
			TerminationSchemaVersion: contracts.APIVersion,
			Termination:              termination,
			AbortID:                  abortID,
			Deadline:                 deadline,
			Reason:                   runstore.Reason{Code: "stage_" + string(termination.Outcome)},
		},
	)
	cancelTransition()
	if err != nil {
		return err
	}
	execution.State = runstore.StageAborting
	execution.TerminationSchemaVersion = stringPointer(contracts.APIVersion)
	execution.Termination = &termination
	execution.AbortID = &abortID
	execution.AbortDeadline = &deadline
	return s.resumeAborting(ctx, run, workflow, execution, reservations)
}

func (s *Scheduler) fenceRecordedAllocations(
	ctx context.Context,
	stageExecutionID string,
	reservations []controlplane.Reservation,
) error {
	seen := make(map[string]struct{}, len(reservations))
	var failures []error
	for _, reservation := range reservations {
		allocationID := reservation.Grant.AllocationID
		seen[allocationID] = struct{}{}
		if err := s.allocator.SetWriteFence(allocationID); err != nil &&
			!errors.Is(err, controlplane.ErrAllocationNotFound) {
			failures = append(failures, err)
		}
	}
	allocations, err := s.store.ListStageAllocations(ctx, stageExecutionID)
	if err != nil {
		failures = append(failures, err)
	} else {
		for _, allocation := range allocations {
			if _, ok := seen[allocation.AllocationID]; ok {
				continue
			}
			if _, err := s.allocator.GetGrant(allocation.AllocationID); errors.Is(err, controlplane.ErrAllocationNotFound) {
				continue
			} else if err != nil {
				failures = append(failures, err)
				continue
			}
			if err := s.allocator.SetWriteFence(allocation.AllocationID); err != nil &&
				!errors.Is(err, controlplane.ErrAllocationNotFound) {
				failures = append(failures, err)
			}
		}
	}
	return errors.Join(failures...)
}

func (s *Scheduler) resumeAborting(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
) error {
	if execution.Termination == nil || execution.AbortID == nil || execution.AbortDeadline == nil {
		return s.failInvalidRunState(ctx, run.RunID, fmt.Errorf("aborting StageExecution is incomplete"))
	}
	if reservations == nil && len(workflow.stage.Agents) > 0 {
		reservations = s.existingLiveReservations(ctx, run, workflow, execution)
	}
	fenceContext, cancelFence := s.terminalOperationContext(ctx)
	_ = s.fenceRecordedAllocations(fenceContext, execution.StageExecutionID, reservations)
	cancelFence()
	var reports map[string]contracts.AllocationFinalReport
	if len(reservations) > 0 && execution.AbortDeadline.After(s.now()) {
		abortContext, cancelAbort := context.WithDeadline(ctx, *execution.AbortDeadline)
		var err error
		reports, err = s.workers.AbortAll(
			abortContext,
			reservations,
			*execution.AbortID,
			contracts.TerminationError{
				Code:      execution.Termination.Code,
				Message:   execution.Termination.Message,
				Retryable: execution.Termination.Retryable,
			},
			*execution.AbortDeadline,
		)
		cancelAbort()
		if err != nil {
			s.options.Logger.Warn("bounded allocation abort was incomplete", "stage_execution_id", execution.StageExecutionID)
		}
	}
	s.persistReports(ctx, execution, reservations, reports)
	commitContext, cancelCommit := s.terminalOperationContext(ctx)
	current, err := s.store.GetRun(commitContext, run.RunID)
	if err != nil {
		cancelCommit()
		return err
	}
	if current.State == runstore.RunCancelling {
		err = s.persistence.CommitTerminationAndFinishRun(
			commitContext,
			run.RunID,
			execution.StageExecutionID,
			runstore.RunCancelling,
			runstore.RunCancelled,
			runstore.Reason{Code: runstore.CancellationUserRequested},
		)
		cancelCommit()
		if err != nil {
			return err
		}
		_ = s.releaseTerminal(execution.StageExecutionID, reservations)
		return nil
	}
	progression, err := s.buildProgression(
		commitContext,
		run,
		workflow,
		execution,
		workflow.stage.On.Interrupted,
		execution.Termination.Retryable,
		execution.Termination.Code,
		runstore.StageExecutionConfigInterruptedEscalation,
	)
	if err == nil {
		err = s.persistence.CommitTerminationProgression(commitContext, TerminationProgression{
			RunID: run.RunID, StageExecutionID: execution.StageExecutionID, Progression: progression,
		})
	}
	cancelCommit()
	if errors.Is(err, runstore.ErrConflict) {
		stateContext, cancelState := s.terminalOperationContext(ctx)
		latest, loadErr := s.store.GetRun(stateContext, run.RunID)
		cancelState()
		if loadErr == nil && latest.State == runstore.RunCancelling {
			return s.resumeAborting(ctx, latest, workflow, execution, reservations)
		}
	}
	if errors.Is(err, runstore.ErrQueuePaused) {
		_ = s.releaseTerminal(execution.StageExecutionID, reservations)
	}
	if err != nil {
		return err
	}
	_ = s.releaseTerminal(execution.StageExecutionID, reservations)
	return nil
}
