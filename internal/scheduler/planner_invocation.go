package scheduler

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func (s *Scheduler) invokeStagePlanner(ctx context.Context, run runstore.WorkflowRun, workflow executableWorkflow, execution runstore.StageExecution, stageDeadline time.Time, prepared *preparedStageWorkers) error {
	reservations, handles := prepared.reservations, prepared.handles
	modelAccess, err := s.plannerModelAccess(ctx, workflow.stage)
	if err != nil {
		return s.beginAbort(ctx, run, workflow, execution, reservations, planner.Failure{
			Code: "planner_execution_config_unavailable", Message: "Planner execution configuration is unavailable", Retryable: false,
		})
	}
	if modelAccess != nil && s.options.GatewayRecovery != nil {
		route := plannerModelRoute(run.OwnerID, workflow.stage)
		modelAccess.Recovery = s.options.GatewayRecovery.Planner(run.RunID, execution.StageExecutionID, *route)
	}
	plannerRef := workflow.stage.Planner.PlannerID + "@" + workflow.stage.Planner.Version
	plannerTelemetry := s.newPlannerTelemetry(
		ctx, run, execution, reservations, plannerRef, modelAccess,
	)
	if plannerTelemetry != nil {
		defer plannerTelemetry.Close()
	}
	instrumentation := telemetry.NoopPlannerInstrumentation()
	if plannerTelemetry != nil {
		instrumentation = plannerTelemetry.Instrumentation()
	}
	invocation := planner.Invocation{
		StageExecutionID: execution.StageExecutionID,
		RunID:            run.RunID,
		Stage:            workflow.stage,
		Context:          plannerContext(execution.StageContext),
		Workers:          handles,
		ModelAccess:      modelAccess,
		Instrumentation:  instrumentation,
		Deadline:         stageDeadline,
	}
	if run.SchedulerClaim != nil {
		invocation.SchedulerClaimID = run.SchedulerClaim.ClaimID
	}
	invocationSpan := instrumentation.StartSpan(
		telemetry.PlannerSpanInvocation,
		telemetry.PlannerSpanAttributes{Operation: "planner.run"},
	)
	instance, err := s.planners.Create(plannerRef, invocation)
	telemetry.CapturePlannerInput(invocationSpan, func() any {
		return map[string]any{"objective": invocation.Stage.Objective, "context": invocation.Context}
	})
	if err != nil {
		invocationSpan.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: "planner_initialization_failed"})
		s.flushPlannerTelemetry(ctx, plannerTelemetry, stageDeadline, execution.StageExecutionID)
		return s.beginAbort(ctx, run, workflow, execution, reservations, planner.Failure{
			Code: "planner_initialization_failed", Message: "Planner could not be initialized", Retryable: false,
		})
	}
	candidate, err := instance.Run(ctx)
	if err == nil {
		telemetry.CapturePlannerOutput(invocationSpan, func() any { return candidate })
	}
	invocationOutcome := "succeeded"
	invocationAttributes := telemetry.PlannerSpanAttributes{}
	if err != nil {
		failure := planner.FailureFrom(err)
		invocationOutcome = "failed"
		invocationAttributes.ErrorCode = failure.Code
	}
	invocationSpan.End(invocationOutcome, invocationAttributes)
	exportResult := s.flushPlannerTelemetry(
		ctx, plannerTelemetry, stageDeadline, execution.StageExecutionID,
	)
	if cause := context.Cause(ctx); errors.Is(cause, ErrAllocationLeaseLost) {
		return cause
	}
	currentExecution, loadErr := s.store.GetStageExecution(ctx, execution.StageExecutionID)
	if loadErr != nil {
		return errors.Join(err, loadErr)
	}
	execution = currentExecution
	s.persistPlannerReport(ctx, execution, instance, exportResult)
	if err != nil {
		return s.beginAbort(ctx, run, workflow, execution, reservations, planner.FailureFrom(err))
	}

	return s.enterFinalizing(ctx, run, workflow, execution, reservations, candidate)
}

func plannerContext(snapshot runstore.StageContextSnapshot) planner.StageContext {
	result := planner.StageContext{
		Parameters: cloneParameters(snapshot.Parameters),
		Artifacts:  make(map[string]*contracts.ArtifactRef, len(snapshot.Artifacts)),
	}
	for name, pinned := range snapshot.Artifacts {
		if pinned.Artifact != nil {
			ref := cloneArtifactRef(*pinned.Artifact)
			result.Artifacts[name] = &ref
		} else {
			result.Artifacts[name] = nil
		}
	}
	return result
}
