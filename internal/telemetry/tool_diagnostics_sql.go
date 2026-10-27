package telemetry

// toolDiagnosticReportSQL authorizes the exact attempt through its owning Run
// and extracts only the selected retained participant. The text-size guard runs
// before a report can leave PostgreSQL; other reports and physical identities
// never cross this read boundary. Eval-held StageMetrics remains authoritative
// after the original execution-report rows expire.
const toolDiagnosticReportSQL = `
WITH owned AS (
    SELECT stage.terminal_at IS NOT NULL AS terminal,
           metrics.metrics_schema_version, metrics.updated_at,
           metrics.xmin::text AS tuple_generation,
           CASE WHEN $4 = 'planner' THEN metrics.metrics->'planner'
                ELSE metrics.metrics->'workers'->$5 END AS report
    FROM stage_executions stage
    JOIN workflow_runs run USING (run_id)
    LEFT JOIN stage_metrics metrics USING (stage_execution_id)
    WHERE run.owner_id = $1 AND run.run_id = $2
      AND stage.stage_execution_id = $3
)
SELECT terminal, metrics_schema_version, updated_at, tuple_generation,
       jsonb_typeof(report),
       COALESCE(octet_length(report::text) > $6, false),
       CASE WHEN octet_length(report::text) <= $6 THEN report END
FROM owned
`
