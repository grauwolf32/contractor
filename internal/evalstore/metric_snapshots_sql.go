package evalstore

// SQL statements for metric_snapshots.go.

// metricSnapshotsQuery returns one row per stage execution of the owner's Runs
// in $1, ordered by (run_id, stage_execution_id) and limited to $3: a redacted
// stage_metrics document (planner/worker counters, completeness flags; NULL
// without metrics), the stage's planner ID/version and its allocation count as
// expected workers. A document is NULL once the running byte total exceeds $4
// or its own size exceeds $5. Used by Store.MetricSnapshots.
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
        COALESCE(stage.stage_spec_snapshot #>> '{planner,plannerId}', '') AS planner_id,
        COALESCE(stage.stage_spec_snapshot #>> '{planner,version}', '') AS planner_version,
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
    expected_workers, planner_id, planner_version
FROM bounded
ORDER BY run_id, stage_execution_id
`
