package runservice

import (
	"context"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runcost"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// CostReports composes the existing diagnostic receipt and in-row cost state.
// Every owning store writes only its records, inside one replay-safe transaction.
type CostReports struct {
	*telemetry.Repository
	pool *pgxpool.Pool
}

func NewCostReports(pool *pgxpool.Pool) *CostReports {
	return &CostReports{Repository: telemetry.NewRepository(pool), pool: pool}
}

func (s *CostReports) RecordAllocationReport(ctx context.Context, e telemetry.AllocationReportEnvelope) error {
	return persistencepostgres.Within(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		contribution, err := telemetry.NewRepository(tx).RecordAllocationReportWithCost(ctx, e)
		if err != nil {
			return err
		}
		return runcost.ApplyReport(ctx, tx, runstore.LLMCostReport{StageExecutionID: e.StageExecutionID, AllocationID: e.AllocationID, Contribution: contribution})
	})
}

func (s *CostReports) RecordPlannerReport(ctx context.Context, e telemetry.PlannerReportEnvelope) error {
	return persistencepostgres.Within(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		contribution, err := telemetry.NewRepository(tx).RecordPlannerReportWithCost(ctx, e)
		if err != nil {
			return err
		}
		return runcost.ApplyReport(ctx, tx, runstore.LLMCostReport{StageExecutionID: e.StageExecutionID, Contribution: contribution})
	})
}
