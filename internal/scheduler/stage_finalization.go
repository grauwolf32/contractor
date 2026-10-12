package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/clone"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// enterFinalizing validates and fences before the atomic candidate transition.
func (s *Scheduler) enterFinalizing(ctx context.Context, run runstore.WorkflowRun, workflow executableWorkflow, execution runstore.StageExecution, reservations []controlplane.Reservation, candidate stageresult.StageContentResult) error {
	if err := s.validateCandidate(ctx, run.RunID, workflow, candidate); err != nil {
		failure := planner.Failure{
			Code: "result_contract_violation", Message: "Planner candidate violates the Stage result contract", Retryable: false,
		}
		var unavailable *candidateVerificationError
		if errors.As(err, &unavailable) {
			failure = planner.Failure{
				Code: "result_verification_unavailable", Message: "Stage result artifact metadata could not be verified", Retryable: true,
			}
		}
		return s.beginAbort(ctx, run, workflow, execution, reservations, failure)
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
	clonedCandidate := candidate.Clone()
	execution.CandidateResult = &clonedCandidate
	execution.FinalizationID = &finalizationID
	execution.FinalizationDeadline = &deadline
	return s.resumeFinalizing(ctx, run, workflow, execution, reservations)

}

type candidateVerificationError struct{ cause error }

func (e *candidateVerificationError) Error() string { return e.cause.Error() }
func (e *candidateVerificationError) Unwrap() error { return e.cause }

func (s *Scheduler) validateCandidate(
	ctx context.Context,
	runID string,
	workflow executableWorkflow,
	result stageresult.StageContentResult,
) error {
	stage := workflow.stage
	if err := result.Validate(); err != nil {
		return err
	}
	for name := range result.Artifacts {
		if _, declared := stage.Result.Artifacts[name]; !declared {
			return fmt.Errorf("undeclared Stage result artifact %q", name)
		}
	}
	if result.Outcome == stageresult.StageSucceeded {
		for name, slot := range stage.Result.Artifacts {
			if _, present := result.Artifacts[name]; slot.Required && !present {
				return fmt.Errorf("required Stage result artifact %q is missing", name)
			}
		}
	}
	names := sortedArtifactNames(result.Artifacts)
	mediaTypes := make(map[string]string, len(names))
	for _, name := range names {
		ref := result.Artifacts[name]
		if artifactpolicy.IsReservedMemoryBinding(ref.Namespace, ref.Name) {
			return fmt.Errorf("Stage result artifact %q identifies a reserved Memory binding", name)
		}
		resolved, err := s.artifacts.Resolve(ctx, runID, ref)
		if err != nil {
			wrapped := fmt.Errorf("verify Stage result artifact %q: %w", name, err)
			if errors.Is(err, artifacts.ErrArtifactNotFound) || errors.Is(err, artifacts.ErrInvalidName) {
				return wrapped
			}
			return &candidateVerificationError{cause: wrapped}
		}
		if !ref.SameExact(resolved.Ref) ||
			!contracts.AcceptsMediaType(stage.Result.Artifacts[name].MediaTypes, resolved.MediaType) {
			return fmt.Errorf("Stage result artifact %q differs from its declared exact version or media type", name)
		}
		mediaTypes[name] = resolved.MediaType
	}
	if result.Outcome == stageresult.StageSucceeded {
		outputNames := make([]string, 0, len(stage.WorkflowOutputs))
		for outputName := range stage.WorkflowOutputs {
			outputNames = append(outputNames, outputName)
		}
		sort.Strings(outputNames)
		for _, outputName := range outputNames {
			resultName := stage.WorkflowOutputs[outputName]
			slot, declared := workflow.workflow.Outputs[outputName]
			if !declared {
				return fmt.Errorf("mapped Workflow output %q is undeclared", outputName)
			}
			mediaType, present := mediaTypes[resultName]
			if !present {
				if slot.Required {
					return fmt.Errorf("required Workflow output %q has no Stage result", outputName)
				}
				continue
			}
			if !contracts.AcceptsMediaType(slot.MediaTypes, mediaType) {
				return fmt.Errorf("Workflow output %q has incompatible media type", outputName)
			}
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
	var reports map[string]reporting.AllocationFinalReport
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
	s.persistReports(ctx, run, workflow, execution, reservations, reports)

	stateContext, cancelState := s.terminalOperationContext(ctx)
	state, err := s.store.GetRunState(stateContext, run.RunID)
	cancelState()
	if err != nil {
		return err
	}
	if state == runstore.RunCancelling {
		return s.acceptFinalizingDuringCancellation(ctx, run, execution, reservations)
	}
	action := workflow.stage.On.Succeeded
	retryable := false
	reasonCode := "workflow_succeeded"
	escalationVariant := runstore.StageExecutionConfigBase
	if execution.CandidateResult.Outcome == stageresult.StageFailed {
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
			Result:          execution.CandidateResult.Clone(),
			WorkflowOutputs: clone.Map(workflow.stage.WorkflowOutputs),
			OutputContracts: resolved.CloneArtifactSlots(workflow.workflow.Outputs),
			Progression:     progression,
		})
	}
	cancelCommit()
	if errors.Is(err, errInvalidRunState) {
		// No replay can progress this candidate, such as an escalation
		// history that the budget cannot count or a mapping onto an output
		// the Run already bound at another revision (ART-R90): end the Run
		// as durable state the Scheduler cannot progress (SCH-R102).
		return s.failInvalidRunState(ctx, run.RunID, err)
	}
	if errors.Is(err, errTransientOutputPublication) {
		err = errors.Join(err, s.countTransientOutputPublication(ctx, execution))
	}
	if errors.Is(err, runstore.ErrConflict) {
		stateContext, cancelState := s.terminalOperationContext(ctx)
		state, loadErr := s.store.GetRunState(stateContext, run.RunID)
		cancelState()
		if loadErr == nil && state == runstore.RunCancelling {
			return s.acceptFinalizingDuringCancellation(ctx, run, execution, reservations)
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

// countTransientOutputPublication records, as its own store operation, a
// terminal commit that a transient publication failure aborted, so that
// finalization recovery stops attempting publication after LIM-PRJ-4 such
// failures (PRJ-R58). A failure it cannot count, because the database is
// unavailable or the claim was lost, does not consume the bound. A finalizing
// Stage that has since left finalizing needs no count.
func (s *Scheduler) countTransientOutputPublication(ctx context.Context, execution runstore.StageExecution) error {
	operation, cancel := s.terminalOperationContext(ctx)
	defer cancel()
	err := s.store.RecordStageOutputPublicationFailure(operation, execution.RunID, execution.StageExecutionID)
	if errors.Is(err, runstore.ErrConflict) {
		return nil
	}
	return err
}

// buildProgression decides the progression of execution's outcome under
// action, the Transition the Run's pinned Workflow declares for it. A check
// of that durable state, the Transition policy, the Stages it names or the
// escalation history its budget counts, fails as errInvalidRunState, which no
// replay changes; a store failure, such as reading the history, stays an
// ordinary error that a later pass retries.
func (s *Scheduler) buildProgression(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	action resolved.TransitionAction,
	retryable bool,
	reasonCode string,
	escalationVariant runstore.StageExecutionConfigVariant,
) (StageProgression, error) {
	selected := action
	var nextEscalationOrdinal *int
	var exhaustedEscalationOrdinal *int
	if action.Kind == resolved.TransitionRetry {
		if action.Retry == nil {
			return StageProgression{}, fmt.Errorf("%w: retry Transition has no bounded policy", errInvalidRunState)
		}
		if !retryable || execution.Attempt >= action.Retry.MaxAttempts {
			selected = action.Retry.Then
		}
	}
	if action.Kind == resolved.TransitionEscalate {
		if action.Escalate == nil ||
			(escalationVariant != runstore.StageExecutionConfigFailedEscalation &&
				escalationVariant != runstore.StageExecutionConfigInterruptedEscalation) {
			return StageProgression{}, fmt.Errorf(
				"%w: escalate Transition has no bounded policy or outcome identity", errInvalidRunState,
			)
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
	case resolved.TransitionRetry:
		targetWorkflow, err := workflow.selectStage(execution.StageName)
		if err != nil {
			return StageProgression{}, fmt.Errorf("%w: %w", errInvalidRunState, err)
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
	case resolved.TransitionEscalate:
		if action.Escalate == nil || nextEscalationOrdinal == nil {
			return StageProgression{}, fmt.Errorf("%w: escalate Transition selection is incomplete", errInvalidRunState)
		}
		targetWorkflow, err := workflow.selectStage(execution.StageName)
		if err != nil {
			return StageProgression{}, fmt.Errorf("%w: %w", errInvalidRunState, err)
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
	case resolved.TransitionNext:
		targetWorkflow, err := workflow.selectStage(selected.NextStage)
		if err != nil {
			return StageProgression{}, fmt.Errorf("%w: %w", errInvalidRunState, err)
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
	case resolved.TransitionSucceed:
		decision.Action = runstore.StageTransitionSucceed
		return StageProgression{
			Decision: decision, TerminalRunState: runstore.RunSucceeded,
			RunReason: runstore.Reason{Code: "workflow_succeeded"},
		}, nil
	case resolved.TransitionFail:
		if strings.TrimSpace(reasonCode) == "" {
			reasonCode = "stage_failed"
		}
		decision.Action = runstore.StageTransitionFail
		return StageProgression{
			Decision: decision, TerminalRunState: runstore.RunFailed,
			RunReason: runstore.Reason{Code: reasonCode},
		}, nil
	default:
		return StageProgression{}, fmt.Errorf("%w: unknown Workflow Transition action %q", errInvalidRunState, selected.Kind)
	}
}

// escalationAttempts counts the automatic escalation attempts of stageName's
// variant branch in the Run's Stage history (SCH-R89). A history the
// Scheduler cannot have written, such as an ordinal that is missing,
// duplicated or not contiguous or a manual continuation that does not repeat
// its source, is errInvalidRunState; a failed read of the history is a store
// error and stays retryable.
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
			return 0, fmt.Errorf("%w: persisted escalation attempt has an invalid ordinal", errInvalidRunState)
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
				return 0, fmt.Errorf(
					"%w: persisted manual escalation continuation has invalid lineage", errInvalidRunState,
				)
			}
			// A manual continuation repeats the pinned configuration; only the
			// original automatic attempt consumes an escalation budget position.
			continue
		}
		ordinal := *current.EscalationOrdinal
		if _, duplicate := seen[ordinal]; duplicate {
			return 0, fmt.Errorf(
				"%w: persisted escalation attempts have duplicate ordinal %d", errInvalidRunState, ordinal,
			)
		}
		seen[ordinal] = struct{}{}
		if ordinal > maxOrdinal {
			maxOrdinal = ordinal
		}
	}
	for ordinal := 1; ordinal <= maxOrdinal; ordinal++ {
		if _, present := seen[ordinal]; !present {
			return 0, fmt.Errorf("%w: persisted escalation attempts have a non-contiguous ordinal", errInvalidRunState)
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
		execution.CandidateResult.Clone(),
	)
	cancelCommit()
	if err != nil {
		return err
	}
	_ = s.releaseTerminal(execution.StageExecutionID, reservations)
	return nil
}
