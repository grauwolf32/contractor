package public

import (
	"context"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// Existing handler fakes have no monetary producer; they still enforce the
// same owner/resource visibility as the real cost reader, and never invent $0.
type testCostReader struct{ services testServices }

func (s testCostReader) GetLLMCost(ctx context.Context, owner, scope, id string) (reporting.LLMCostSummary, error) {
	switch scope {
	case "run":
		value, err := s.services.Runs.GetRun(ctx, id)
		if err != nil || value.OwnerID != owner {
			return reporting.LLMCostSummary{}, telemetry.ErrNotFound
		}
	case "audit":
		if _, err := s.services.Audits.Get(ctx, owner, id); err != nil {
			return reporting.LLMCostSummary{}, telemetry.ErrNotFound
		}
	case "project":
		if _, err := s.services.Projects.Get(ctx, owner, id); err != nil {
			return reporting.LLMCostSummary{}, telemetry.ErrNotFound
		}
	default:
		return reporting.LLMCostSummary{}, telemetry.ErrInvalid
	}
	return reporting.BuildLLMCostSummary(scope, id, reporting.NewLLMCostTotals(true), false, true, time.Date(2026, 11, 8, 0, 0, 0, 0, time.UTC)), nil
}

func (s testCostReader) GetAuditExecutionCosts(ctx context.Context, owner, auditID string, ids []string) ([]reporting.LLMCostExecution, error) {
	if _, err := s.GetLLMCost(ctx, owner, "audit", auditID); err != nil {
		return nil, err
	}
	items := []reporting.LLMCostExecution{}
	for _, id := range ids {
		items = append(items, reporting.LLMCostExecution{ExecutionID: id, Basis: "none", Coverage: "unavailable"})
	}
	return items, nil
}
