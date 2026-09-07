package runstore

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/telemetry"
)

func (s *PostgresStore) RecordStageExecutionReport(
	ctx context.Context,
	params RecordStageExecutionReportParams,
) error {
	err := telemetry.NewRepository(s.db).RecordAllocationReport(
		ctx,
		telemetry.AllocationReportEnvelope{
			StageExecutionID:            params.StageExecutionID,
			AllocationID:                params.AllocationID,
			LogicalAgentName:            params.LogicalAgentName,
			ReportSchemaVersion:         params.ReportSchemaVersion,
			Report:                      params.Report,
			PerformanceCollectionPolicy: params.PerformanceCollectionPolicy,
			Secrets:                     params.Secrets,
		},
	)
	return mapTelemetryError("record StageExecution report", err)
}

func (s *PostgresStore) ListStageExecutionReports(
	ctx context.Context,
	stageExecutionID string,
) ([]StageExecutionReport, error) {
	rows, err := telemetry.NewRepository(s.db).ListAllocationReports(ctx, stageExecutionID)
	if err != nil {
		return nil, mapTelemetryError("list StageExecution reports", err)
	}
	result := make([]StageExecutionReport, 0, len(rows))
	for _, item := range rows {
		result = append(result, StageExecutionReport{
			StageExecutionID:    item.StageExecutionID,
			AllocationID:        item.AllocationID,
			LogicalAgentName:    item.LogicalAgentName,
			ReportSchemaVersion: item.ReportSchemaVersion,
			Report:              item.Report,
			ReceivedAt:          item.ReceivedAt,
			ExpiresAt:           item.ExpiresAt,
		})
	}
	return result, nil
}

func (s *PostgresStore) RecordPlannerExecutionReport(
	ctx context.Context,
	params RecordPlannerExecutionReportParams,
) error {
	err := telemetry.NewRepository(s.db).RecordPlannerReport(
		ctx,
		telemetry.PlannerReportEnvelope{
			StageExecutionID:    params.StageExecutionID,
			SessionID:           params.SessionID,
			InvocationID:        params.InvocationID,
			StartedAt:           params.StartedAt,
			FinishedAt:          params.FinishedAt,
			ReportSchemaVersion: params.ReportSchemaVersion,
			Report:              params.Report,
			Secrets:             params.Secrets,
		},
	)
	return mapTelemetryError("record Planner execution report", err)
}

func (s *PostgresStore) RebuildStageMetrics(
	ctx context.Context,
	stageExecutionID string,
	schemaVersion string,
) error {
	_, err := telemetry.NewRepository(s.db).RebuildStageMetrics(
		ctx, stageExecutionID, schemaVersion,
	)
	return mapTelemetryError("rebuild StageMetrics", err)
}

func (s *PostgresStore) CleanupExpiredTelemetry(
	ctx context.Context,
	now time.Time,
	batchSize int,
) (int64, error) {
	deleted, err := telemetry.NewRepository(s.db).CleanupExpired(ctx, now, batchSize)
	if err != nil {
		return deleted, mapTelemetryError("cleanup expired telemetry", err)
	}
	return deleted, nil
}

func mapTelemetryError(operation string, err error) error {
	if err == nil {
		return nil
	}
	switch {
	case errors.Is(err, telemetry.ErrInvalid):
		return fmt.Errorf("%s: %w", operation, ErrInvalid)
	case errors.Is(err, telemetry.ErrConflict):
		return fmt.Errorf("%s: %w", operation, ErrConflict)
	case errors.Is(err, telemetry.ErrNotFound):
		return fmt.Errorf("%s: %w", operation, ErrNotFound)
	default:
		return fmt.Errorf("%s: %w", operation, err)
	}
}
