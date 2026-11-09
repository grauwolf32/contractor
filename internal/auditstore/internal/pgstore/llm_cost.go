package pgstore

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/jackc/pgx/v5"
)

// RunCosts is the transaction-bound seam for ordinary Run report attribution.
// It exposes no Audit lifecycle or controller commands.
type RunCosts struct{ store }

func NewRunCostsTx(tx pgx.Tx) *RunCosts { return &RunCosts{store: store{db: tx}} }

func (s *RunCosts) Apply(ctx context.Context, executionID, runID, role string, old, next *reporting.CostContribution, runTotal reporting.LLMCostTotals) error {
	var auditID string
	var raw []byte
	if err := s.db.QueryRow(ctx, lockAuditCostSQL, executionID, runID).Scan(&auditID, &raw); err != nil {
		return err
	}
	total, err := reporting.DecodeLLMCostTotals(raw)
	if err != nil {
		return err
	}
	if err := total.ReplaceCostContribution(old, next, role, true, next.ReceivedAt); err != nil {
		return err
	}
	encoded, err := total.Encode()
	if err != nil {
		return err
	}
	if _, err = s.db.Exec(ctx, updateAuditCostSQL, auditID, encoded); err != nil {
		return err
	}
	encoded, err = runTotal.Encode()
	if err != nil {
		return err
	}
	_, err = s.db.Exec(ctx, updateAuditExecutionCostSQL, executionID, encoded)
	return err
}

func (s *QueryStore) ReadExecutionCosts(ctx context.Context, ownerID, auditID string, ids []string) ([]reporting.LLMCostExecution, error) {
	if len(ids) == 0 || len(ids) > 64 {
		return nil, auditstore.ErrInvalid
	}
	rows, err := s.db.Query(ctx, readAuditExecutionCostsSQL, ownerID, auditID, ids)
	if err != nil {
		return nil, err
	}
	items, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (reporting.LLMCostExecution, error) {
		var item reporting.LLMCostExecution
		var raw []byte
		var pending bool
		if err := row.Scan(&item.ExecutionID, &item.RunID, &raw, &pending); err != nil {
			return item, err
		}
		total, err := reporting.DecodeLLMCostTotals(raw)
		if err != nil {
			return item, err
		}
		value := reporting.BuildLLMCostSummary("run", item.ExecutionID, total, pending, true, time.Now().UTC())
		item.Amount, item.Basis, item.Coverage, item.Pending = value.Amount, value.Basis, value.Coverage, value.Pending
		return item, nil
	})
	if err == nil && len(items) != len(ids) {
		return nil, auditstore.ErrNotFound
	}
	return items, err
}

func (s *QueryStore) ReadLLMCost(ctx context.Context, ownerID, auditID string) (reporting.LLMCostTotals, bool, bool, error) {
	var raw []byte
	var pending, hasWork bool
	err := s.db.QueryRow(ctx, readAuditCostSQL, ownerID, auditID).Scan(&raw, &pending, &hasWork)
	if errors.Is(err, pgx.ErrNoRows) {
		return reporting.LLMCostTotals{}, false, false, auditstore.ErrNotFound
	}
	if err != nil {
		return reporting.LLMCostTotals{}, false, false, err
	}
	total, err := reporting.DecodeLLMCostTotals(raw)
	return total, pending, hasWork, err
}
