package evalstore

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// MetricSnapshot projects only counters/completeness. It excludes prompts,
// tool arguments, errors and physical artifact locations.
type MetricSnapshot struct {
	RunID            string
	StageExecutionID string
	Document         json.RawMessage
	ExpectedWorkers  int
}

const metricSnapshotsQuery = `
WITH snapshots AS (
    SELECT stage.run_id, stage.stage_execution_id,
        CASE WHEN metrics.stage_execution_id IS NOT NULL THEN jsonb_build_object(
            'planner', CASE WHEN metrics.metrics -> 'planner' IS NOT NULL
                AND metrics.metrics -> 'planner' <> 'null'::jsonb
                THEN jsonb_build_object(
                    'complete', metrics.metrics #> '{planner,complete}',
                    'truncated', metrics.metrics #> '{planner,truncated}',
                    'metrics', metrics.metrics #> '{planner,metrics}'
                )
            END,
            'workers', COALESCE((
                SELECT jsonb_object_agg(key, jsonb_build_object(
                    'complete', value -> 'complete',
                    'truncated', value -> 'truncated',
                    'metrics', value -> 'metrics'
                ))
                FROM jsonb_each(metrics.metrics -> 'workers')
            ), '{}'::jsonb),
            'runtime', COALESCE((
                SELECT jsonb_object_agg(key, jsonb_build_object('complete', value -> 'complete'))
                FROM jsonb_each(metrics.metrics -> 'runtime')
            ), '{}'::jsonb)
        ) END AS document,
        (SELECT count(*) FROM stage_allocations allocation
            WHERE allocation.stage_execution_id = stage.stage_execution_id)::int AS expected_workers
    FROM stage_executions stage
    JOIN workflow_runs run USING (run_id)
    LEFT JOIN stage_metrics metrics USING (stage_execution_id)
    WHERE stage.run_id = ANY($1) AND run.owner_id = $2
    ORDER BY stage.run_id, stage.stage_execution_id
    LIMIT $3
), bounded AS (
    SELECT *, sum(COALESCE(octet_length(document::text), 0))
        OVER (ORDER BY run_id, stage_execution_id) AS total_size
    FROM snapshots
)
SELECT run_id, stage_execution_id,
    CASE WHEN total_size <= $4 AND octet_length(document::text) <= $5 THEN document END,
    expected_workers
FROM bounded
ORDER BY run_id, stage_execution_id
`

// The caller supplies the Run IDs from its authoritative member inventory.
// Ownership is checked again here. One extra row exposes truncation explicitly.
func (s *Store) MetricSnapshots(ctx context.Context, owner string, runs []string) ([]MetricSnapshot, error) {
	if len(runs) > evaldomain.MaxInventoryExecutions {
		return nil, evaldomain.Failure("eval_limit_exceeded")
	}
	rows, err := s.db.Query(ctx, metricSnapshotsQuery, runs, owner, evaldomain.MaxMetricSnapshots+1, evaldomain.MaxCollectionBytes, evaldomain.MaxDocumentBytes)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	snapshots := []MetricSnapshot{}
	for rows.Next() {
		var item MetricSnapshot
		if err = rows.Scan(&item.RunID, &item.StageExecutionID, &item.Document, &item.ExpectedWorkers); err != nil {
			return nil, err
		}
		snapshots = append(snapshots, item)
	}
	return snapshots, rows.Err()
}
