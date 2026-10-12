package runstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/telemetry"
)

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
