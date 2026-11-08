package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/jackc/pgx/v5"
)

type LLMCostReport struct {
	StageExecutionID string
	AllocationID     string // Empty identifies the Stage's single Planner.
	Contribution     reporting.CostContribution
}

type LLMCostChange struct {
	RunID, OwnerID, Role        string
	ProjectID, AuditExecutionID *string
	Previous                    *reporting.CostContribution
	Next                        reporting.CostContribution
	RunTotal                    reporting.LLMCostTotals
	Changed                     bool
}

// ApplyLLMCostReport runs inside the report composition's transaction. The
// existing Run row serializes its participants; their last counted snapshots
// survive expiring diagnostic reports without another receipt table.
func (s *PostgresStore) ApplyLLMCostReport(ctx context.Context, p LLMCostReport) (LLMCostChange, error) {
	result := LLMCostChange{Next: p.Contribution, Role: "planner"}
	if _, ok := s.db.(pgx.Tx); !ok {
		return result, ErrInvalid
	}
	if err := p.Contribution.Validate(); err != nil {
		return result, err
	}
	var total, prior []byte
	err := s.db.QueryRow(ctx, costRunContextSQL, p.StageExecutionID).Scan(&result.RunID, &result.OwnerID, &result.ProjectID, &result.AuditExecutionID, &total)
	if errors.Is(err, pgx.ErrNoRows) {
		return result, ErrNotFound
	}
	if err != nil {
		return result, err
	}
	query := costPlannerContributionSQL
	args := []any{p.StageExecutionID}
	if p.AllocationID != "" {
		result.Role = "worker"
		query = costAllocationContributionSQL
		args = append(args, p.AllocationID)
	}
	if err := s.db.QueryRow(ctx, query, args...).Scan(&prior); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return result, ErrNotFound
		}
		return result, err
	}
	result.Previous, err = reporting.DecodeCostContribution(prior)
	if err != nil {
		return result, fmt.Errorf("decode counted cost: %w", err)
	}
	result.Changed, err = reporting.CostContributionChanged(result.Previous, p.Contribution)
	if err != nil {
		return result, fmt.Errorf("%w: invalid cost replay", ErrConflict)
	}
	if !result.Changed {
		return result, nil
	}
	result.RunTotal, err = reporting.DecodeLLMCostTotals(total)
	if err != nil {
		return result, err
	}
	if err := result.RunTotal.ReplaceCostContribution(result.Previous, &result.Next, result.Role, result.AuditExecutionID != nil, result.Next.ReceivedAt); err != nil {
		return result, err
	}
	encoded, err := json.Marshal(result.Next)
	if err != nil {
		return result, err
	}
	if p.AllocationID == "" {
		_, err = s.db.Exec(ctx, updatePlannerCostContributionSQL, p.StageExecutionID, encoded)
	} else {
		_, err = s.db.Exec(ctx, updateAllocationCostContributionSQL, p.StageExecutionID, p.AllocationID, encoded)
	}
	if err != nil {
		return result, err
	}
	encoded, err = result.RunTotal.Encode()
	if err != nil {
		return result, err
	}
	_, err = s.db.Exec(ctx, updateRunCostSQL, result.RunID, encoded)
	return result, err
}

func (s *PostgresStore) ReadLLMCost(ctx context.Context, ownerID, runID string) (reporting.LLMCostTotals, bool, bool, error) {
	var raw []byte
	var pending, hasWork bool
	err := s.db.QueryRow(ctx, readRunCostSQL, ownerID, runID).Scan(&raw, &pending, &hasWork)
	if errors.Is(err, pgx.ErrNoRows) {
		return reporting.LLMCostTotals{}, false, false, ErrNotFound
	}
	if err != nil {
		return reporting.LLMCostTotals{}, false, false, err
	}
	value, err := reporting.DecodeLLMCostTotals(raw)
	return value, pending, hasWork, err
}
