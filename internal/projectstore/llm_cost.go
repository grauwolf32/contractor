package projectstore

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/jackc/pgx/v5"
)

// ApplyLLMCost adds one participant replacement delta to the existing Project.
// The caller owns the report/Run/Audit/Project transaction.
func (s *PostgresStore) ApplyLLMCost(ctx context.Context, ownerID, projectID, role string, inCheck bool, old, next *reporting.CostContribution) error {
	if _, ok := s.db.(pgx.Tx); !ok {
		return ErrInvalid
	}
	var raw []byte
	if err := s.db.QueryRow(ctx, lockProjectCostSQL, ownerID, projectID).Scan(&raw); err != nil {
		return err
	}
	total, err := reporting.DecodeLLMCostTotals(raw)
	if err != nil {
		return err
	}
	if err := total.ReplaceCostContribution(old, next, role, inCheck, next.ReceivedAt); err != nil {
		return err
	}
	encoded, err := total.Encode()
	if err != nil {
		return err
	}
	_, err = s.db.Exec(ctx, updateProjectCostSQL, projectID, encoded)
	return err
}

func (s *PostgresStore) ReadLLMCost(ctx context.Context, ownerID, projectID string) (reporting.LLMCostTotals, bool, bool, error) {
	var raw []byte
	var pending, hasWork bool
	err := s.db.QueryRow(ctx, readProjectCostSQL, ownerID, projectID).Scan(&raw, &pending, &hasWork)
	if errors.Is(err, pgx.ErrNoRows) {
		return reporting.LLMCostTotals{}, false, false, ErrNotFound
	}
	if err != nil {
		return reporting.LLMCostTotals{}, false, false, err
	}
	total, err := reporting.DecodeLLMCostTotals(raw)
	return total, pending, hasWork, err
}
