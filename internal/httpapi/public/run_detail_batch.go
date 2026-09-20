package public

import (
	"context"

	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

type runDetailRelated struct {
	allocations map[string][]runstore.StageAllocation
	metrics     map[string]telemetry.StageMetricsRecord
	plans       map[string]planner.PlannerPlanProjection
	resources   map[string][]telemetry.AllocationResourceSummary
}

// Call only with executions of an already owner-authorized Run. Production
// PostgreSQL dependencies issue at most one query per collection, including
// stages with no allocations or optional metrics. All readers must implement
// the batch contract.
func (h *handler) loadRunDetailRelated(ctx context.Context, ownerID string, executions []runstore.StageExecution) (runDetailRelated, error) {
	result := runDetailRelated{
		allocations: make(map[string][]runstore.StageAllocation),
		metrics:     make(map[string]telemetry.StageMetricsRecord),
		plans:       make(map[string]planner.PlannerPlanProjection),
		resources:   make(map[string][]telemetry.AllocationResourceSummary),
	}
	ids := make([]string, len(executions))
	var identities []planner.SessionIdentity
	for i, execution := range executions {
		ids[i] = execution.StageExecutionID
		if execution.PlannerSessionID != nil && execution.PlannerInvocationID != nil {
			identities = append(identities, planner.SessionIdentity{SessionID: *execution.PlannerSessionID, StageExecutionID: execution.StageExecutionID, InvocationID: *execution.PlannerInvocationID})
		}
	}
	if len(ids) == 0 {
		return result, nil
	}
	var err error
	result.allocations, err = h.dependencies.Runs.ListStageAllocationsBatch(ctx, ids)
	if err != nil {
		return result, err
	}
	if h.dependencies.Metrics != nil {
		// Optional diagnostics must not prevent reading the durable Run state.
		result.metrics, _ = h.dependencies.Metrics.GetStageMetricsBatch(ctx, ids)
	}
	if h.dependencies.PlannerPlans != nil {
		result.plans, err = h.dependencies.PlannerPlans.LoadPlans(ctx, identities)
		if err != nil {
			return result, err
		}
	}
	result.resources, err = h.dependencies.AllocationResources.ListStageAllocationResources(ctx, ownerID, ids)
	if err != nil {
		return result, err
	}
	return result, nil
}
