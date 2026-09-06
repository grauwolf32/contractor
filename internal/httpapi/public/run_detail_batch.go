package public

import (
	"context"

	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

type allocationBatchReader interface {
	ListStageAllocationsBatch(context.Context, []string) (map[string][]runstore.StageAllocation, error)
}
type metricsBatchReader interface {
	GetStageMetricsBatch(context.Context, []string) (map[string]telemetry.StageMetricsRecord, error)
}
type planBatchReader interface {
	LoadPlans(context.Context, []planner.SessionIdentity) (map[string]planner.PlannerPlanProjection, error)
}

type runDetailRelated struct {
	allocations map[string][]runstore.StageAllocation
	metrics     map[string]telemetry.StageMetricsRecord
	plans       map[string]planner.PlannerPlanProjection
}

// Call only with executions of an already owner-authorized Run. Production
// PostgreSQL dependencies issue at most one query per collection, including
// stages with no allocations or optional metrics. Compatibility fallbacks are
// for non-PostgreSQL implementations, not concurrent per-stage query fan-out.
func (h *handler) loadRunDetailRelated(ctx context.Context, executions []runstore.StageExecution) (runDetailRelated, error) {
	result := runDetailRelated{
		allocations: make(map[string][]runstore.StageAllocation),
		metrics:     make(map[string]telemetry.StageMetricsRecord),
		plans:       make(map[string]planner.PlannerPlanProjection),
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
	if batch, ok := h.dependencies.Runs.(allocationBatchReader); ok {
		result.allocations, err = batch.ListStageAllocationsBatch(ctx, ids)
		if err != nil {
			return result, err
		}
	} else {
		for _, id := range ids {
			result.allocations[id], err = h.dependencies.Runs.ListStageAllocations(ctx, id)
			if err != nil {
				return result, err
			}
		}
	}
	if batch, ok := h.dependencies.Metrics.(metricsBatchReader); ok {
		// Metrics remain best effort; never fall back to N queries on failure.
		result.metrics, _ = batch.GetStageMetricsBatch(ctx, ids)
	} else if h.dependencies.Metrics != nil {
		for _, id := range ids {
			if record, err := h.dependencies.Metrics.GetStageMetrics(ctx, id); err == nil {
				result.metrics[id] = record
			}
		}
	}
	if batch, ok := h.dependencies.PlannerPlans.(planBatchReader); ok {
		result.plans, err = batch.LoadPlans(ctx, identities)
		if err != nil {
			return result, err
		}
	} else if h.dependencies.PlannerPlans != nil {
		for _, identity := range identities {
			plan, present, err := h.dependencies.PlannerPlans.LoadPlan(ctx, identity)
			if err != nil {
				return result, err
			}
			if present {
				result.plans[identity.StageExecutionID] = plan
			}
		}
	}
	return result, nil
}
