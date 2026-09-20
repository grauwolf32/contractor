package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// enterFinalizing validates and fences before the atomic candidate transition.
func (s *Scheduler) enterFinalizing(ctx context.Context, run runstore.WorkflowRun, workflow executableWorkflow, execution runstore.StageExecution, reservations []controlplane.Reservation, candidate contracts.StageContentResult) error {
	if err := s.validateCandidate(ctx, run.RunID, workflow.stage, candidate); err != nil {
		return s.beginAbort(ctx, run, workflow, execution, reservations, planner.Failure{
			Code: "result_contract_violation", Message: "Planner candidate violates the Stage result contract", Retryable: false,
		})
	}
	if err := s.fenceRecordedAllocations(ctx, execution.StageExecutionID, reservations); err != nil {
		return s.beginAbort(ctx, run, workflow, execution, reservations, planner.Failure{
			Code: "allocation_fence_failed", Message: "Stage allocations could not be write-fenced", Retryable: true,
		})
	}
	finalizationID, err := s.newID("finalization_")
	if err != nil {
		return err
	}
	deadline := s.now().Add(s.options.FinalizationTimeout)
	transitionContext, cancelTransition := context.WithTimeout(ctx, s.options.OperationTimeout)
	err = s.persistence.EnterFinalizingWithResult(transitionContext, runstore.EnterFinalizingParams{
		StageExecutionID:    execution.StageExecutionID,
		ResultSchemaVersion: contracts.APIVersion,
		Candidate:           candidate,
		FinalizationID:      finalizationID,
		Deadline:            deadline,
		Reason:              runstore.Reason{Code: "planner_completed"},
	})
	cancelTransition()
	if err != nil {
		return err
	}
	execution.State = runstore.StageFinalizing
	execution.CandidateResultSchemaVersion = stringPointer(contracts.APIVersion)
	clonedCandidate := cloneStageResult(candidate)
	execution.CandidateResult = &clonedCandidate
	execution.FinalizationID = &finalizationID
	execution.FinalizationDeadline = &deadline
	return s.resumeFinalizing(ctx, run, workflow, execution, reservations)

}

func (s *Scheduler) validateCandidate(
	ctx context.Context,
	runID string,
	stage workflowconfig.ResolvedStage,
	result contracts.StageContentResult,
) error {
	if err := result.Validate(); err != nil {
		return err
	}
	encoded, err := json.Marshal(result)
	if err != nil || len(encoded) > contracts.MaxStageResultBytes || len(result.Summary) > contracts.MaxStageResultSummaryBytes ||
		len(result.Artifacts) > contracts.MaxStageResultArtifacts {
		return fmt.Errorf("StageResult exceeds its bounded contract")
	}
	for name := range result.Artifacts {
		if _, declared := stage.Result.Artifacts[name]; !declared {
			return fmt.Errorf("undeclared Stage result artifact %q", name)
		}
	}
	if result.Outcome == contracts.StageSucceeded {
		for name, slot := range stage.Result.Artifacts {
			if _, present := result.Artifacts[name]; slot.Required && !present {
				return fmt.Errorf("required Stage result artifact %q is missing", name)
			}
		}
	}
	names := sortedArtifactNames(result.Artifacts)
	for _, name := range names {
		ref := result.Artifacts[name]
		if artifactpolicy.IsReservedMemoryBinding(ref.Namespace, ref.Name) {
			return fmt.Errorf("Stage result artifact %q identifies a reserved Memory binding", name)
		}
		resolved, err := s.artifacts.Resolve(ctx, runID, ref)
		if err != nil {
			return fmt.Errorf("verify Stage result artifact %q: %w", name, err)
		}
		if !sameExactRef(ref, resolved.Ref) ||
			!acceptsMediaType(stage.Result.Artifacts[name].MediaTypes, resolved.MediaType) {
			return fmt.Errorf("Stage result artifact %q differs from its declared exact version or media type", name)
		}
	}
	return nil
}

func (s *Scheduler) resumeFinalizing(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
) error {
	if execution.CandidateResult == nil || execution.FinalizationID == nil ||
		execution.FinalizationDeadline == nil {
		return s.failInvalidRunState(ctx, run.RunID, fmt.Errorf("finalizing StageExecution is incomplete"))
	}
	if reservations == nil && len(workflow.stage.Agents) > 0 {
		reservations = s.existingLiveReservations(ctx, run, workflow, execution)
	}
	var reports map[string]contracts.AllocationFinalReport
	if len(reservations) > 0 && execution.FinalizationDeadline.After(s.now()) {
		finalizeContext, cancelFinalize := context.WithDeadline(ctx, *execution.FinalizationDeadline)
		var err error
		reports, err = s.workers.FinalizeAll(
			finalizeContext,
			reservations,
			*execution.FinalizationID,
			*execution.FinalizationDeadline,
		)
		cancelFinalize()
		if err != nil {
			s.options.Logger.Warn("bounded allocation finalization was incomplete", "stage_execution_id", execution.StageExecutionID)
		}
	}
	s.persistReports(ctx, execution, reservations, reports)

	stateContext, cancelState := s.terminalOperationContext(ctx)
	current, err := s.store.GetRun(stateContext, run.RunID)
	cancelState()
	if err != nil {
		return err
	}
	if current.State == runstore.RunCancelling {
		return s.acceptFinalizingDuringCancellation(ctx, current, execution, reservations)
	}
	action := workflow.stage.On.Succeeded
	retryable := false
	reasonCode := "workflow_succeeded"
	escalationVariant := runstore.StageExecutionConfigBase
	if execution.CandidateResult.Outcome == contracts.StageFailed {
		action = workflow.stage.On.Failed
		retryable = execution.CandidateResult.Error != nil && execution.CandidateResult.Error.Retryable
		escalationVariant = runstore.StageExecutionConfigFailedEscalation
		reasonCode = "stage_failed"
		if execution.CandidateResult.Error != nil {
			reasonCode = execution.CandidateResult.Error.Code
		}
	}
	commitContext, cancelCommit := s.terminalOperationContext(ctx)
	progression, err := s.buildProgression(
		commitContext, run, workflow, execution, action, retryable, reasonCode, escalationVariant,
	)
	if err == nil {
		err = s.persistence.CommitResultProgression(commitContext, ResultProgression{
			RunID: run.RunID, StageExecutionID: execution.StageExecutionID,
			Result:          cloneStageResult(*execution.CandidateResult),
			WorkflowOutputs: cloneStringMap(workflow.stage.WorkflowOutputs),
			OutputContracts: cloneArtifactSlots(workflow.workflow.Outputs),
			Progression:     progression,
		})
	}
	cancelCommit()
	if errors.Is(err, runstore.ErrConflict) {
		stateContext, cancelState := s.terminalOperationContext(ctx)
		current, loadErr := s.store.GetRun(stateContext, run.RunID)
		cancelState()
		if loadErr == nil && current.State == runstore.RunCancelling {
			return s.acceptFinalizingDuringCancellation(ctx, current, execution, reservations)
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

func (s *Scheduler) buildProgression(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	action workflowconfig.TransitionAction,
	retryable bool,
	reasonCode string,
	escalationVariant runstore.StageExecutionConfigVariant,
) (StageProgression, error) {
	selected := action
	var nextEscalationOrdinal *int
	var exhaustedEscalationOrdinal *int
	if action.Kind == workflowconfig.TransitionRetry {
		if action.Retry == nil {
			return StageProgression{}, fmt.Errorf("retry Transition has no bounded policy")
		}
		if !retryable || execution.Attempt >= action.Retry.MaxAttempts {
			selected = action.Retry.Then
		}
	}
	if action.Kind == workflowconfig.TransitionEscalate {
		if action.Escalate == nil ||
			(escalationVariant != runstore.StageExecutionConfigFailedEscalation &&
				escalationVariant != runstore.StageExecutionConfigInterruptedEscalation) {
			return StageProgression{}, fmt.Errorf("escalate Transition has no bounded policy or outcome identity")
		}
		used, err := s.escalationAttempts(ctx, run.RunID, execution.StageName, escalationVariant)
		if err != nil {
			return StageProgression{}, err
		}
		if used >= action.Escalate.MaxAttempts {
			selected = action.Escalate.Then
			exhaustedEscalationOrdinal = intPointer(used)
		} else {
			nextEscalationOrdinal = intPointer(used + 1)
		}
	}
	decision := runstore.RecordStageTransitionDecisionParams{
		SourceExecutionID: execution.StageExecutionID,
		RunID:             run.RunID,
	}
	if exhaustedEscalationOrdinal != nil {
		decision.EscalationOrdinal = exhaustedEscalationOrdinal
		decision.EscalationExhausted = true
	}
	switch selected.Kind {
	case workflowconfig.TransitionRetry:
		targetWorkflow, err := workflow.selectStage(execution.StageName)
		if err != nil {
			return StageProgression{}, err
		}
		previous := execution.StageExecutionID
		creation, err := s.buildStageCreation(
			ctx, run, targetWorkflow, execution.Attempt+1, &previous,
			stageExecutionConfiguration{variant: runstore.StageExecutionConfigBase},
		)
		if err != nil {
			return StageProgression{}, err
		}
		targetStage, targetExecution := creation.Params.StageName, creation.Params.StageExecutionID
		decision.Action = runstore.StageTransitionRetry
		decision.TargetStageName = &targetStage
		decision.TargetExecutionID = &targetExecution
		return StageProgression{Decision: decision, NextStage: &creation}, nil
	case workflowconfig.TransitionEscalate:
		if action.Escalate == nil || nextEscalationOrdinal == nil {
			return StageProgression{}, fmt.Errorf("escalate Transition selection is incomplete")
		}
		targetWorkflow, err := workflow.selectStage(execution.StageName)
		if err != nil {
			return StageProgression{}, err
		}
		previous := execution.StageExecutionID
		effective := action.Escalate.ExecutionConfig.Effective
		creation, err := s.buildStageCreation(
			ctx, run, targetWorkflow, execution.Attempt+1, &previous,
			stageExecutionConfiguration{
				variant: escalationVariant, ordinal: nextEscalationOrdinal, effective: &effective,
			},
		)
		if err != nil {
			return StageProgression{}, err
		}
		targetStage, targetExecution := creation.Params.StageName, creation.Params.StageExecutionID
		decision.Action = runstore.StageTransitionEscalate
		decision.TargetStageName = &targetStage
		decision.TargetExecutionID = &targetExecution
		decision.EscalationOrdinal = nextEscalationOrdinal
		return StageProgression{Decision: decision, NextStage: &creation}, nil
	case workflowconfig.TransitionNext:
		targetWorkflow, err := workflow.selectStage(selected.NextStage)
		if err != nil {
			return StageProgression{}, err
		}
		creation, err := s.buildStageCreation(
			ctx, run, targetWorkflow, 1, nil,
			stageExecutionConfiguration{variant: runstore.StageExecutionConfigBase},
		)
		if err != nil {
			return StageProgression{}, err
		}
		targetStage, targetExecution := creation.Params.StageName, creation.Params.StageExecutionID
		decision.Action = runstore.StageTransitionNext
		decision.TargetStageName = &targetStage
		decision.TargetExecutionID = &targetExecution
		return StageProgression{Decision: decision, NextStage: &creation}, nil
	case workflowconfig.TransitionSucceed:
		decision.Action = runstore.StageTransitionSucceed
		return StageProgression{
			Decision: decision, TerminalRunState: runstore.RunSucceeded,
			RunReason: runstore.Reason{Code: "workflow_succeeded"},
		}, nil
	case workflowconfig.TransitionFail:
		if strings.TrimSpace(reasonCode) == "" {
			reasonCode = "stage_failed"
		}
		decision.Action = runstore.StageTransitionFail
		return StageProgression{
			Decision: decision, TerminalRunState: runstore.RunFailed,
			RunReason: runstore.Reason{Code: reasonCode},
		}, nil
	default:
		return StageProgression{}, fmt.Errorf("unknown Workflow Transition action %q", selected.Kind)
	}
}

func (s *Scheduler) escalationAttempts(
	ctx context.Context,
	runID string,
	stageName string,
	variant runstore.StageExecutionConfigVariant,
) (int, error) {
	executions, err := s.store.ListStageExecutions(ctx, runID)
	if err != nil {
		return 0, err
	}
	byID := make(map[string]runstore.StageExecution, len(executions))
	for _, execution := range executions {
		byID[execution.StageExecutionID] = execution
	}
	maxOrdinal := 0
	seen := make(map[int]struct{})
	for _, current := range executions {
		if current.StageName != stageName || current.ExecutionConfigVariant != variant {
			continue
		}
		if current.EscalationOrdinal == nil || *current.EscalationOrdinal <= 0 {
			return 0, fmt.Errorf("persisted escalation attempt has an invalid ordinal")
		}
		if current.ResumeSourceExecutionID != nil {
			previous, exists := byID[*current.ResumeSourceExecutionID]
			if !exists || current.PreviousExecutionID == nil ||
				*current.PreviousExecutionID != previous.StageExecutionID ||
				previous.RunID != current.RunID || previous.StageName != current.StageName ||
				previous.Attempt+1 != current.Attempt ||
				previous.ExecutionConfigVariant != current.ExecutionConfigVariant ||
				previous.EscalationOrdinal == nil || *previous.EscalationOrdinal != *current.EscalationOrdinal ||
				(previous.State != runstore.StageFailed && previous.State != runstore.StageInterrupted) {
				return 0, fmt.Errorf("persisted manual escalation continuation has invalid lineage")
			}
			// A manual continuation repeats the pinned configuration; only the
			// original automatic attempt consumes an escalation budget position.
			continue
		}
		ordinal := *current.EscalationOrdinal
		if _, duplicate := seen[ordinal]; duplicate {
			return 0, fmt.Errorf("persisted escalation attempts have duplicate ordinal %d", ordinal)
		}
		seen[ordinal] = struct{}{}
		if ordinal > maxOrdinal {
			maxOrdinal = ordinal
		}
	}
	for ordinal := 1; ordinal <= maxOrdinal; ordinal++ {
		if _, present := seen[ordinal]; !present {
			return 0, fmt.Errorf("persisted escalation attempts have a non-contiguous ordinal")
		}
	}
	return maxOrdinal, nil
}

func (s *Scheduler) acceptFinalizingDuringCancellation(
	ctx context.Context,
	run runstore.WorkflowRun,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
) error {
	commitContext, cancelCommit := s.terminalOperationContext(ctx)
	err := s.persistence.AcceptResultDuringCancellation(
		commitContext,
		run.RunID,
		execution.StageExecutionID,
		cloneStageResult(*execution.CandidateResult),
	)
	cancelCommit()
	if err != nil {
		return err
	}
	_ = s.releaseTerminal(execution.StageExecutionID, reservations)
	return nil
}

func (s *Scheduler) finishRunFromTerminalTermination(
	ctx context.Context,
	run runstore.WorkflowRun,
	execution runstore.StageExecution,
) error {
	if execution.Termination == nil {
		return s.failInvalidRunState(ctx, run.RunID, fmt.Errorf("terminal interrupted Stage has no termination"))
	}
	return s.failActiveRun(ctx, run.RunID, execution.Termination.Code, nil)
}
