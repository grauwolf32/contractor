package telemetry

// SQL statements for repository.go.

// listAllocationReportsSQL returns one effective allocation execution report
// per logical agent of StageExecution $1: a report whose worker and runtime
// sections are both complete wins, then the newest, then the lowest report_id.
// Used by Repository.ListAllocationReports.
var listAllocationReportsSQL = `
WITH effective AS (
    SELECT DISTINCT ON (logical_agent_name)
           stage_execution_id, allocation_id, logical_agent_name,
           report_schema_version, report, received_at, expires_at
    FROM allocation_execution_reports
    WHERE stage_execution_id = $1
    ORDER BY logical_agent_name,
             ((report->'worker'->>'complete')::boolean
              AND (report->'runtime'->>'complete')::boolean) DESC,
             received_at DESC, report_id
)
SELECT stage_execution_id, allocation_id, logical_agent_name,
       report_schema_version, report, received_at, expires_at
FROM effective ORDER BY logical_agent_name`

// rebuildStageMetricsSQL upserts the computed metrics and summary of
// StageExecution $1 and refreshes updated_at. On conflict expires_at keeps the
// later of the stored value and the column default, so retention never
// shrinks. Returns the stored row. Used by Repository.RebuildStageMetrics.
var rebuildStageMetricsSQL = `
INSERT INTO stage_metrics (
    stage_execution_id, metrics_schema_version, metrics, summary
) VALUES ($1, $2, $3::jsonb, $4::jsonb)
ON CONFLICT (stage_execution_id) DO UPDATE SET
    metrics_schema_version = EXCLUDED.metrics_schema_version,
    metrics = EXCLUDED.metrics,
    summary = EXCLUDED.summary,
    updated_at = clock_timestamp(),
    expires_at = GREATEST(stage_metrics.expires_at, EXCLUDED.expires_at)
RETURNING metrics_schema_version, metrics, summary, updated_at, expires_at`

// cleanupStageMetricsSQL settles at most $2 StageMetrics rows that expired by
// $1 and belong to terminal StageExecutions, oldest expiry first. A row of a
// Run that an existing Eval member executed (contractor_eval_run_members) is
// renewed through the column default, one more retention period, instead of
// deleted (OBS-R43). The renewal leaves the expired range, so later passes
// spend their budget on other rows. Returns the number of rows deleted or
// renewed. Used by Repository.CleanupExpired.
var cleanupStageMetricsSQL = `
WITH expired AS (
    SELECT m.stage_execution_id, e.run_id FROM stage_metrics m
    JOIN stage_executions e USING (stage_execution_id)
    WHERE m.expires_at <= $1
      AND e.state IN ('succeeded', 'failed', 'interrupted', 'cancelled')
    ORDER BY m.expires_at, m.stage_execution_id LIMIT $2
    FOR UPDATE OF m SKIP LOCKED
), classified AS (
    SELECT stage_execution_id,
           EXISTS (SELECT 1 FROM contractor_eval_run_members(run_id)) AS retained
    FROM expired
), renewed AS (
    UPDATE stage_metrics m SET expires_at = DEFAULT
    FROM classified c
    WHERE m.stage_execution_id = c.stage_execution_id AND c.retained
    RETURNING 1
), deleted AS (
    DELETE FROM stage_metrics m USING classified c
    WHERE m.stage_execution_id = c.stage_execution_id AND NOT c.retained
    RETURNING 1
)
SELECT (SELECT count(*) FROM renewed) + (SELECT count(*) FROM deleted)`
