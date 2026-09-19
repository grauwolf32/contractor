package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func (s *Scheduler) liveOrNewReservations(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
) ([]controlplane.Reservation, bool, error) {
	requirements, err := auditBindingRequirements(run, workflow, execution.StageContext)
	if err != nil {
		return nil, false, err
	}
	recorded, err := s.store.ListStageAllocations(ctx, execution.StageExecutionID)
	if err != nil {
		return nil, false, err
	}
	if execution.State == runstore.StageRunning && len(recorded) == 0 {
		return nil, false, errControlPlaneStateLost
	}
	if len(recorded) > 0 {
		lost := false
		for _, allocation := range recorded {
			grant, grantErr := s.allocator.GetGrant(allocation.AllocationID)
			if grantErr != nil || grant.StageExecutionID != execution.StageExecutionID {
				return nil, false, errControlPlaneStateLost
			}
			lost = lost || grant.Lost
		}
		if lost {
			reservations, reserveErr := s.reserveAll(ctx, controlplane.ReservationRequest{
				RunID: run.RunID, StageExecutionID: execution.StageExecutionID,
				RunMetadataLabels: run.MetadataLabels.Clone(),
				Bindings:          requirements, RuntimeConfig: &run.RuntimeConfig,
			})
			if reserveErr != nil {
				return nil, false, errControlPlaneAllocationLost
			}
			return reservations, false, errControlPlaneAllocationLost
		}
	}
	reservations, err := s.reserveAll(ctx, controlplane.ReservationRequest{
		RunID: run.RunID, StageExecutionID: execution.StageExecutionID,
		RunMetadataLabels: run.MetadataLabels.Clone(),
		Bindings:          requirements, RuntimeConfig: &run.RuntimeConfig,
	})
	if err != nil {
		return nil, false, err
	}
	if err := verifyReservations(run, workflow, execution, recorded, reservations); err != nil {
		return nil, false, err
	}
	for _, reservation := range reservations {
		if reservation.Grant.Lost {
			return reservations, false, errControlPlaneAllocationLost
		}
	}
	return reservations, len(recorded) == 0, nil
}

func (s *Scheduler) recordReservations(
	ctx context.Context,
	stageExecutionID string,
	reservations []controlplane.Reservation,
) error {
	for _, reservation := range reservations {
		collectionPolicy := reservation.PerformanceCollectionPolicy
		if collectionPolicy.ValidatePinned() != nil {
			// Legacy in-process allocators predate optional resource collection.
			collectionPolicy = contracts.PerformanceCollectionDisabled
		}
		allocation := runstore.StageAllocation{
			CompletionContract: contracts.CloneWorkerCompletionContract(reservation.CompletionContract),
			AllocationID:       reservation.Grant.AllocationID, StageExecutionID: stageExecutionID,
			LogicalAgentName: reservation.Grant.LogicalAgentName, Namespace: reservation.Grant.Namespace,
			AgentTemplateRef:            reservation.AgentTemplate.Ref,
			WorkerRuntimeRef:            reservation.AgentTemplate.Runtime,
			RuntimeAgentID:              reservation.Grant.RuntimeAgentID,
			RuntimeAgentInstanceID:      reservation.Grant.RuntimeInstanceID,
			RuntimeAgentLabelRevision:   reservation.RuntimeAgentLabelRevision,
			PerformanceCollectionPolicy: collectionPolicy,
		}
		if reservation.ResolvedRuntimeConfig != nil {
			resolved := reservation.ResolvedRuntimeConfig
			allocation.RuntimeConfigurationSchemaVersion = runstore.AllocationRuntimeConfigurationSchemaVersion
			allocation.RuntimeConfiguration = &runstore.AllocationRuntimeConfiguration{
				ModelPolicy: resolved.ModelPolicy.Ref,
				Origins:     resolved.Origins,
				Provenance:  resolved.Provenance,
			}
		}
		operationContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
		err := s.store.RecordStageAllocation(operationContext, allocation)
		cancel()
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *Scheduler) releaseUnprepared(reservations []controlplane.Reservation) {
	for _, reservation := range reservations {
		if err := s.allocator.Release(reservation.Grant.AllocationID); err != nil {
			s.options.Logger.Warn("release unprepared allocation failed", "allocation_id", reservation.Grant.AllocationID)
		}
	}
}

func verifyReservations(
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	recorded []runstore.StageAllocation,
	reservations []controlplane.Reservation,
) error {
	expectedBindings, err := auditBindingRequirements(run, workflow, execution.StageContext)
	if err != nil {
		return err
	}
	expectedContracts := make(map[string]*contracts.WorkerCompletionContract, len(expectedBindings))
	for _, binding := range expectedBindings {
		expectedContracts[binding.LogicalAgentName] = binding.CompletionContract
	}
	expectedWorkspace, err := projectAllocationWorkspace(workflow.stage.Context.Workspace, execution.StageContext)
	if err != nil {
		return err
	}
	if len(reservations) != len(workflow.stage.Agents) {
		return fmt.Errorf("Control Plane returned an incomplete allocation set")
	}
	recordedByName := make(map[string]runstore.StageAllocation, len(recorded))
	for _, allocation := range recorded {
		recordedByName[allocation.LogicalAgentName] = allocation
	}
	seen := make(map[string]struct{}, len(reservations))
	for _, reservation := range reservations {
		grant := reservation.Grant
		binding, ok := workflow.stage.Agents[grant.LogicalAgentName]
		if !ok || grant.RunID != run.RunID || grant.StageExecutionID != execution.StageExecutionID ||
			grant.Namespace != binding.Namespace || reservation.AgentTemplate.Ref != binding.Template.Ref ||
			reservation.WorkerSessionMode != workflow.stage.Session ||
			reservation.AgentTemplate.Runtime != binding.Template.Runtime || reservation.LeaseExpiresAt.IsZero() ||
			!reflect.DeepEqual(reservation.Workspace, expectedWorkspace) ||
			!equalRunMetadataLabels(reservation.RunMetadataLabels, run.MetadataLabels) ||
			!reflect.DeepEqual(reservation.CompletionContract, expectedContracts[grant.LogicalAgentName]) {
			return fmt.Errorf("Control Plane returned an allocation for different resolved inputs")
		}
		if reservation.ResolvedRuntimeConfig == nil {
			if !sameAllocationExecutionConfig(
				reservation.ExecutionConfig,
				allocationExecutionConfig(workflow.stage, grant.LogicalAgentName),
			) {
				return fmt.Errorf("Control Plane returned an allocation for different execution config")
			}
		} else {
			resolved := reservation.ResolvedRuntimeConfig
			selection := workflow.stage.ExecutionConfig.Agents[grant.LogicalAgentName]
			if resolved.Validate() != nil || resolved.ModelFree != binding.Template.IsToolWorker() || resolved.ModelPolicy.Ref != selection.ModelPolicy.Ref ||
				reservation.RuntimeAgentLabelRevision == 0 || grant.RuntimeAgentID == "" ||
				reservation.PerformanceCollectionPolicy.ValidatePinned() != nil ||
				(reservation.PerformanceCollectionPolicy == contracts.PerformanceCollectionRequested) !=
					(reservation.PerformanceMetrics != nil) ||
				(reservation.PerformanceMetrics != nil && reservation.PerformanceMetrics.Validate() != nil) ||
				!sameAllocationExecutionConfig(reservation.ExecutionConfig, controlplane.AllocationExecutionConfig{
					ModelPolicy: resolved.ModelPolicy.Ref, LLMGateway: resolved.LLMGateway.Ref,
					Credential: resolved.LLMCredential,
				}) {
				return fmt.Errorf("Control Plane returned invalid candidate Runtime provenance")
			}
		}
		if _, duplicate := seen[grant.LogicalAgentName]; duplicate {
			return fmt.Errorf("Control Plane returned duplicate logical Agent allocations")
		}
		seen[grant.LogicalAgentName] = struct{}{}
		if persisted, exists := recordedByName[grant.LogicalAgentName]; exists &&
			(!reflect.DeepEqual(persisted.CompletionContract, reservation.CompletionContract) || persisted.AllocationID != grant.AllocationID ||
				persisted.RuntimeAgentInstanceID != grant.RuntimeInstanceID ||
				persisted.Namespace != grant.Namespace || persisted.AgentTemplateRef != binding.Template.Ref ||
				persisted.WorkerRuntimeRef != binding.Template.Runtime ||
				reservation.ResolvedRuntimeConfig != nil &&
					(persisted.RuntimeAgentID != grant.RuntimeAgentID ||
						persisted.RuntimeAgentLabelRevision != reservation.RuntimeAgentLabelRevision ||
						persisted.PerformanceCollectionPolicy != reservation.PerformanceCollectionPolicy ||
						persisted.RuntimeConfigurationSchemaVersion != runstore.AllocationRuntimeConfigurationSchemaVersion ||
						!samePersistedRuntimeConfiguration(persisted.RuntimeConfiguration, reservation.ResolvedRuntimeConfig))) {
			return fmt.Errorf("live Control Plane allocation differs from durable provenance")
		}
	}
	if len(recorded) > 0 && len(recordedByName) != len(seen) {
		return fmt.Errorf("durable allocation set is incomplete")
	}
	return nil
}

func equalRunMetadataLabels(left, right map[string]string) bool {
	if len(left) != len(right) {
		return false
	}
	for key, value := range left {
		other, present := right[key]
		if !present || other != value {
			return false
		}
	}
	return true
}

func samePersistedRuntimeConfiguration(
	persisted *runstore.AllocationRuntimeConfiguration,
	resolved *runtimeconfig.ResolvedRuntimeConfig,
) bool {
	if persisted == nil || resolved == nil {
		return persisted == nil && resolved == nil
	}
	want := runstore.AllocationRuntimeConfiguration{
		ModelPolicy: resolved.ModelPolicy.Ref, Origins: resolved.Origins, Provenance: resolved.Provenance,
	}
	left, leftErr := json.Marshal(persisted)
	right, rightErr := json.Marshal(want)
	return leftErr == nil && rightErr == nil && string(left) == string(right)
}

func (s *Scheduler) existingLiveReservations(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
) []controlplane.Reservation {
	requirements, err := auditBindingRequirements(run, workflow, execution.StageContext)
	if err != nil {
		return nil
	}
	recorded, err := s.store.ListStageAllocations(ctx, execution.StageExecutionID)
	if err != nil || len(recorded) == 0 {
		return nil
	}
	for _, allocation := range recorded {
		if _, err := s.allocator.GetGrant(allocation.AllocationID); err != nil {
			return nil
		}
	}
	reservations, err := s.reserveAll(ctx, controlplane.ReservationRequest{
		RunID: run.RunID, StageExecutionID: execution.StageExecutionID,
		RunMetadataLabels: run.MetadataLabels.Clone(),
		Bindings:          requirements, RuntimeConfig: &run.RuntimeConfig,
	})
	if err != nil || verifyReservations(run, workflow, execution, recorded, reservations) != nil {
		return nil
	}
	return reservations
}

type contextAllocator interface {
	ReserveAllContext(context.Context, controlplane.ReservationRequest) ([]controlplane.Reservation, error)
}

func (s *Scheduler) reserveAll(
	ctx context.Context,
	request controlplane.ReservationRequest,
) ([]controlplane.Reservation, error) {
	if allocator, ok := s.allocator.(contextAllocator); ok {
		return allocator.ReserveAllContext(ctx, request)
	}
	return s.allocator.ReserveAll(request)
}
