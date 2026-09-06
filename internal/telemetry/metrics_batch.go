package telemetry

import (
	"context"
	"encoding/json"
	"fmt"
)

// GetStageMetricsBatch loads optional read-model diagnostics in one query.
// Missing or undecodable records are omitted individually, matching Run detail's
// best-effort GetStageMetrics behavior without suppressing healthy siblings.
func (r *Repository) GetStageMetricsBatch(ctx context.Context, ids []string) (map[string]StageMetricsRecord, error) {
	result := make(map[string]StageMetricsRecord, len(ids))
	for _, id := range ids {
		if err := requireText("stageExecutionID", id); err != nil {
			return nil, err
		}
	}
	if len(ids) == 0 {
		return result, nil
	}
	rows, err := r.db.Query(ctx, `SELECT stage_execution_id, metrics_schema_version, metrics, summary, updated_at, expires_at
FROM stage_metrics WHERE stage_execution_id = ANY($1::text[])`, ids)
	if err != nil {
		return nil, fmt.Errorf("list StageMetrics: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var record StageMetricsRecord
		var metrics, summary []byte
		if err := rows.Scan(&record.StageExecutionID, &record.MetricsSchemaVersion, &metrics, &summary, &record.UpdatedAt, &record.ExpiresAt); err != nil {
			return nil, fmt.Errorf("scan StageMetrics: %w", err)
		}
		if json.Unmarshal(metrics, &record.Metrics) != nil || json.Unmarshal(summary, &record.Summary) != nil {
			continue
		}
		result[record.StageExecutionID] = record
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate StageMetrics: %w", err)
	}
	return result, nil
}
