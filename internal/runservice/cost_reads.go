package runservice

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func (s *CostReports) GetLLMCost(ctx context.Context, ownerID, scope, id string) (reporting.LLMCostSummary, error) {
	var total reporting.LLMCostTotals
	var pending, hasWork bool
	var err error
	switch scope {
	case "run":
		total, pending, hasWork, err = runstore.NewPostgresStore(s.pool, nil).ReadLLMCost(ctx, ownerID, id)
	case "audit":
		total, pending, hasWork, err = querystore.New(s.pool, nil).ReadLLMCost(ctx, ownerID, id)
	case "project":
		total, pending, hasWork, err = projectstore.NewPostgresStore(s.pool).ReadLLMCost(ctx, ownerID, id)
	default:
		return reporting.LLMCostSummary{}, telemetry.ErrInvalid
	}
	if errors.Is(err, runstore.ErrNotFound) || errors.Is(err, auditstore.ErrNotFound) || errors.Is(err, projectstore.ErrNotFound) {
		return reporting.LLMCostSummary{}, telemetry.ErrNotFound
	}
	if err != nil {
		return reporting.LLMCostSummary{}, err
	}
	return reporting.BuildLLMCostSummary(scope, id, total, pending, hasWork, time.Now().UTC()), nil
}

func (s *CostReports) GetAuditExecutionCosts(ctx context.Context, owner, auditID string, ids []string) ([]reporting.LLMCostExecution, error) {
	items, err := querystore.New(s.pool, nil).ReadExecutionCosts(ctx, owner, auditID, ids)
	if errors.Is(err, auditstore.ErrNotFound) {
		return nil, telemetry.ErrNotFound
	}
	if errors.Is(err, auditstore.ErrInvalid) {
		return nil, telemetry.ErrInvalid
	}
	return items, err
}
