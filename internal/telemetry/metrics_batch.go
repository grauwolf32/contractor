package telemetry

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// GetStageMetricsBatch loads optional read-model diagnostics in one query.
// Each record is decoded as strictly as it was validated on write; a missing
// record, or one that fails that decoding, is omitted individually, so Run
// detail stays best-effort without suppressing healthy siblings (RUN-R90).
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
		var metricsErr, summaryErr error
		record.Metrics, metricsErr = decodeStoredStageMetrics(metrics)
		record.Summary, summaryErr = decodeStoredSummary(summary)
		if metricsErr != nil || summaryErr != nil {
			continue
		}
		result[record.StageExecutionID] = record
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate StageMetrics: %w", err)
	}
	return result, nil
}

// decodeStoredStageMetrics decodes a stored StageMetrics document as strictly
// as its reports were decoded when received and applies the rules
// RebuildStageMetrics checked before storing it, except the byte bounds
// (OBS-R110).
func decodeStoredStageMetrics(data []byte) (reporting.StageMetrics, error) {
	var metrics reporting.StageMetrics
	if err := strictjson.DecodeNonNull(data, &metrics); err != nil {
		return reporting.StageMetrics{}, err
	}
	if err := metrics.Validate(); err != nil {
		return reporting.StageMetrics{}, err
	}
	return metrics, nil
}

// decodeStoredSummary decodes a stored Stage summary strictly: every member
// Summarize writes is present and every counter is non-negative (OBS-R112).
func decodeStoredSummary(data []byte) (Summary, error) {
	var summary Summary
	if err := strictjson.DecodeNonNull(data, &summary); err != nil {
		return Summary{}, err
	}
	if err := summary.Validate(); err != nil {
		return Summary{}, err
	}
	return summary, nil
}
