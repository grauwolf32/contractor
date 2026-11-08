package runservice

import (
	"context"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
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
	if e.ReceivedAt.IsZero() {
		e.ReceivedAt = time.Now().UTC()
	}
	return persistencepostgres.Within(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := telemetry.NewRepository(tx).RecordAllocationReport(ctx, e); err != nil {
			return err
		}
		normalized, err := telemetry.NewPolicy(e.Secrets...).NormalizeExecutionReport(e.Report.Worker, telemetry.MaxReportJSONBytes)
		if err != nil {
			return err
		}
		return applyCostReport(ctx, tx, runstore.LLMCostReport{StageExecutionID: e.StageExecutionID, AllocationID: e.AllocationID, Contribution: reportCostContribution(e.Report.ReportID, e.ReceivedAt, normalized)})
	})
}

func (s *CostReports) RecordPlannerReport(ctx context.Context, e telemetry.PlannerReportEnvelope) error {
	if e.ReceivedAt.IsZero() {
		e.ReceivedAt = time.Now().UTC()
	}
	return persistencepostgres.Within(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := telemetry.NewRepository(tx).RecordPlannerReport(ctx, e); err != nil {
			return err
		}
		normalized, err := telemetry.NewPolicy(e.Secrets...).NormalizeExecutionReport(e.Report, telemetry.MaxReportJSONBytes)
		if err != nil {
			return err
		}
		return applyCostReport(ctx, tx, runstore.LLMCostReport{StageExecutionID: e.StageExecutionID, Contribution: reportCostContribution(e.Report.ReportID, e.ReceivedAt, normalized)})
	})
}

func reportCostContribution(id string, received time.Time, report reporting.ExecutionReport) reporting.CostContribution {
	var calls int64
	if report.Metrics.ModelCalls != nil {
		calls = *report.Metrics.ModelCalls
	}
	return reporting.CostContribution{ReportID: id, ReceivedAt: received, Complete: report.Complete, ModelCalls: calls, Cost: report.Metrics.LLMCost}
}

func applyCostReport(ctx context.Context, tx pgx.Tx, p runstore.LLMCostReport) error {
	change, err := runstore.NewPostgresStore(tx, nil).ApplyLLMCostReport(ctx, p)
	if err != nil || !change.Changed {
		return err
	}
	if change.AuditExecutionID != nil {
		if err := hookstore.NewRunCostsTx(tx).Apply(ctx, *change.AuditExecutionID, change.RunID, change.Role, change.Previous, &change.Next, change.RunTotal); err != nil {
			return err
		}
	}
	if change.ProjectID != nil {
		return projectstore.NewPostgresStore(tx).ApplyLLMCost(ctx, change.OwnerID, *change.ProjectID, change.Role, change.AuditExecutionID != nil, change.Previous, &change.Next)
	}
	return nil
}
