package telemetry

// toolDiagnosticReportSQL authorizes the exact attempt through its owning Run
// and extracts only the selected retained participant. The text-size guard runs
// before a report can leave PostgreSQL; other reports and physical identities
// never cross this read boundary. Eval-held StageMetrics remains authoritative
// after the original execution-report rows expire.
// Extract the selected path once, without copying the containing Worker map.
// Keep the selected JSON text materialized so byte checks and the result reuse
// one serialization; neither boundary carries the full StageMetrics document.
const toolDiagnosticReportSQL = `
WITH owned AS MATERIALIZED (
    SELECT stage.terminal_at IS NOT NULL AS terminal,
           metrics.metrics_schema_version, metrics.updated_at,
           metrics.xmin::text AS tuple_generation,
           CASE WHEN $4 = 'planner' THEN metrics.metrics->'planner'
                WHEN metrics.metrics @? 'strict $.workers ? (@.type() == "object")'
                THEN metrics.metrics #> ARRAY['workers', $5] END AS report
    FROM stage_executions stage
    JOIN workflow_runs run USING (run_id)
    LEFT JOIN stage_metrics metrics USING (stage_execution_id)
    WHERE run.owner_id = $1 AND run.run_id = $2
      AND stage.stage_execution_id = $3
), projected AS MATERIALIZED (
    SELECT terminal, metrics_schema_version, updated_at, tuple_generation,
           jsonb_typeof(report) AS report_kind, report::text AS document
    FROM owned
)
SELECT terminal, metrics_schema_version, updated_at, tuple_generation,
       report_kind,
       COALESCE(octet_length(document) > $6, false),
       CASE WHEN octet_length(document) <= $6 THEN document END
FROM projected
`
