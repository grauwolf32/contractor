package scheduler

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (s *Scheduler) createStageExecution(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
) (runstore.StageExecution, error) {
	creation, err := s.buildStageCreation(
		ctx, run, workflow, 1, nil,
		stageExecutionConfiguration{variant: runstore.StageExecutionConfigBase},
	)
	if err != nil {
		return runstore.StageExecution{}, err
	}
	operationContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
	defer cancel()
	return s.persistence.CreateStageWithContext(operationContext, creation.Params, creation.ContextPins)
}

func (s *Scheduler) buildStageCreation(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	attempt int,
	previousExecutionID *string,
	configuration stageExecutionConfiguration,
) (NextStageCreation, error) {
	stageExecutionID, err := s.newID("stage_execution_")
	if err != nil {
		return NextStageCreation{}, fmt.Errorf("generate StageExecution ID: %w", err)
	}
	contextSnapshot := runstore.StageContextSnapshot{
		Parameters: cloneParameters(run.Parameters),
		Artifacts:  make(map[string]runstore.PinnedContextArtifact, len(workflow.stage.Context.Artifacts)),
	}
	pins := make([]ContextPin, 0, len(workflow.stage.Context.Artifacts))
	names := make([]string, 0, len(workflow.stage.Context.Artifacts))
	for name := range workflow.stage.Context.Artifacts {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		declaration := workflow.stage.Context.Artifacts[name]
		current := auditPinnedContextRef(run, contracts.ArtifactRef{Namespace: declaration.Namespace, Name: declaration.Name})
		resolved, resolveErr := s.artifacts.Resolve(ctx, run.RunID, current)
		if errors.Is(resolveErr, artifacts.ErrArtifactNotFound) {
			contextSnapshot.Artifacts[name] = runstore.PinnedContextArtifact{Required: declaration.Required}
			continue
		}
		if resolveErr != nil {
			return NextStageCreation{}, fmt.Errorf("resolve StageContext artifact %q: %w", name, resolveErr)
		}
		if resolved.Ref.Namespace != declaration.Namespace || resolved.Ref.Name != declaration.Name || (current.Revision != nil && !reflect.DeepEqual(current, resolved.Ref)) {
			return NextStageCreation{}, fmt.Errorf("ArtifactStore resolved StageContext artifact %q to another binding", name)
		}
		exact := cloneArtifactRef(resolved.Ref)
		contextSnapshot.Artifacts[name] = runstore.PinnedContextArtifact{
			Required: declaration.Required,
			Artifact: &exact,
		}
		pins = append(pins, ContextPin{Name: name, Ref: exact})
	}
	stage := workflow.stage
	if configuration.variant == "" {
		configuration.variant = runstore.StageExecutionConfigBase
	}
	switch configuration.variant {
	case runstore.StageExecutionConfigBase:
		if configuration.ordinal != nil || configuration.effective != nil {
			return NextStageCreation{}, fmt.Errorf("base Stage execution configuration has escalation data")
		}
	case runstore.StageExecutionConfigFailedEscalation,
		runstore.StageExecutionConfigInterruptedEscalation:
		if configuration.ordinal == nil || *configuration.ordinal <= 0 || configuration.effective == nil {
			return NextStageCreation{}, fmt.Errorf("escalated Stage execution configuration is incomplete")
		}
		stage.ExecutionConfig = *configuration.effective
	default:
		return NextStageCreation{}, fmt.Errorf("unknown Stage execution configuration %q", configuration.variant)
	}
	encodedStage, err := stageSnapshot(stage)
	if err != nil {
		return NextStageCreation{}, err
	}
	return NextStageCreation{Params: runstore.CreateStageExecutionParams{
		StageExecutionID: stageExecutionID,
		RunID:            run.RunID, StageName: workflow.stageName, Attempt: attempt,
		PreviousExecutionID:    previousExecutionID,
		ExecutionConfigVariant: configuration.variant,
		EscalationOrdinal:      configuration.ordinal,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: encodedStage,
		StageContextSchemaVersion: contracts.APIVersion, StageContext: contextSnapshot,
	}, ContextPins: pins}, nil
}

type stageExecutionConfiguration struct {
	variant   runstore.StageExecutionConfigVariant
	ordinal   *int
	effective *workflowconfig.ResolvedStageExecutionConfig
}

func missingRequiredContext(execution runstore.StageExecution) string {
	names := make([]string, 0, len(execution.StageContext.Artifacts))
	for name := range execution.StageContext.Artifacts {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		value := execution.StageContext.Artifacts[name]
		if value.Required && value.Artifact == nil {
			return name
		}
	}
	return ""
}

// prepareAndPlan retains one Stage deadline across allocation and Planner phases.
func (s *Scheduler) prepareAndPlan(ctx context.Context, run runstore.WorkflowRun, workflow executableWorkflow, execution runstore.StageExecution) error {
	deadline := s.stageDeadline(execution)
	prepared, err := s.prepareStageWorkers(ctx, run, workflow, execution, deadline)
	if err != nil || prepared == nil {
		return err
	}
	return s.invokeStagePlanner(ctx, run, workflow, execution, prepared.deadline, prepared)
}

// A nil prepared Stage means a durable abort/defer path has already handled it.
type preparedStageWorkers struct {
	deadline     time.Time
	reservations []controlplane.Reservation
	handles      map[string]contracts.WorkerHandle
}

func (s *Scheduler) prepareStageWorkers(ctx context.Context, run runstore.WorkflowRun, workflow executableWorkflow, execution runstore.StageExecution, stageDeadline time.Time) (*preparedStageWorkers, error) {
	reservations, fresh, err := s.liveOrNewReservations(ctx, run, workflow, execution)
	if errors.Is(err, controlplane.ErrInsufficientCapacity) {
		if execution.AdmittedAt != nil && !s.now().Before(stageDeadline) {
			return nil, s.beginAbort(ctx, run, workflow, execution, nil, stageDeadlineFailure())
		}
		return nil, ErrDeferred
	}
	if errors.Is(err, errControlPlaneStateLost) {
		return nil, s.beginAbort(ctx, run, workflow, execution, nil, planner.Failure{
			Code: "control_plane_state_lost", Message: "Control Plane lost the active allocation set", Retryable: true,
		})
	}
	if errors.Is(err, errControlPlaneAllocationLost) {
		cleanupContext, cancelCleanup := claimCleanupContext(ctx)
		defer cancelCleanup()
		return nil, s.beginAbort(cleanupContext, run, workflow, execution, reservations, planner.Failure{
			Code: "control_lease_expired", Message: "Runtime Agent allocation control lease was lost", Retryable: true,
		})
	}
	if err != nil {
		return nil, err
	}
	if execution.AdmittedAt != nil && !s.now().Before(stageDeadline) {
		return nil, s.beginAbort(ctx, run, workflow, execution, reservations, stageDeadlineFailure())
	}
	if cause := context.Cause(ctx); errors.Is(cause, ErrAllocationLeaseLost) {
		return nil, cause
	}
	if fresh {
		allowed, gateErr := s.admitModelRoutes(ctx, run, workflow.stage, reservations)
		if gateErr != nil || !allowed {
			s.releaseUnprepared(reservations)
			if gateErr != nil {
				return nil, gateErr
			}
			return nil, ErrDeferred
		}
	}
	if execution.AdmittedAt == nil {
		execution, err = s.persistence.AdmitStage(ctx, run.RunID, execution.StageExecutionID)
		if err != nil {
			s.releaseUnprepared(reservations)
			return nil, err
		}
		stageDeadline = s.stageDeadline(execution)
	}
	if fresh {
		if err := s.recordReservations(ctx, execution.StageExecutionID, reservations); err != nil {
			s.releaseUnprepared(reservations)
			return nil, s.beginAbort(ctx, run, workflow, execution, nil, planner.Failure{
				Code: "allocation_record_failed", Message: "Stage allocation provenance could not be recorded", Retryable: true,
			})
		}
	}

	if fresh {
		if err := s.bindModelRoutes(ctx, run, reservations); err != nil {
			return nil, s.beginAbort(ctx, run, workflow, execution, reservations, planner.Failure{Code: "model_route_binding_failed", Message: "Model recovery route could not be bound", Retryable: true})
		}
	}

	workerSettings, err := s.workerExecutionSettingsForRun(ctx, run, workflow.stage, reservations)
	if err != nil {
		return nil, s.beginAbort(ctx, run, workflow, execution, reservations, planner.Failure{
			Code: "worker_execution_config_unavailable", Message: "Worker execution configuration is unavailable", Retryable: false,
		})
	}
	prepareContext, cancelPrepare := context.WithTimeout(ctx, s.options.OperationTimeout)
	handles, err := s.workers.PrepareAll(prepareContext, reservations, workerSettings)
	cancelPrepare()
	clearWorkerExecutionSettings(workerSettings)
	if err != nil {
		failure := infrastructureFailure("allocation_preparation_failed", "Worker allocation preparation failed", err)
		return nil, s.beginAbort(ctx, run, workflow, execution, nil, failure)
	}
	if cause := context.Cause(ctx); errors.Is(cause, ErrAllocationLeaseLost) {
		return nil, cause
	}
	if execution.AdmittedAt != nil && !s.now().Before(stageDeadline) {
		return nil, s.beginAbort(ctx, run, workflow, execution, reservations, stageDeadlineFailure())
	}

	return &preparedStageWorkers{reservations: reservations, handles: handles, deadline: stageDeadline}, nil
}

func (s *Scheduler) stageDeadline(execution runstore.StageExecution) time.Time {
	if execution.AdmittedAt != nil {
		return execution.AdmittedAt.Add(s.options.PlannerTimeout)
	}
	return s.now().Add(s.options.PlannerTimeout)
}

func stageDeadlineFailure() planner.Failure {
	return planner.Failure{
		Code:      "stage_deadline_exceeded",
		Message:   "Stage execution deadline expired before Planner completion",
		Retryable: true,
	}
}
