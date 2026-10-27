package telemetry

// runtimeDiagnosticPageSQL authorizes one exact Stage through its owning Run,
// then projects only a bounded page of Runtime participants from retained
// StageMetrics. Eval holds retain this authority after the source report rows
// expire. The lateral LIMIT applies before values leave PostgreSQL; neither
// Worker reports nor tool-call arguments are returned. The short SQL stopReason
// prefix is additionally bounded by UTF-8 bytes at the public projection.
const runtimeDiagnosticPageSQL = `
WITH owned AS (
    SELECT stage.terminal_at IS NOT NULL AS terminal,
           metrics.metrics_schema_version, metrics.updated_at,
           metrics.xmin::text AS tuple_generation,
           metrics.metrics->'runtime' AS runtime,
           EXISTS (
               SELECT 1 FROM jsonb_object_keys(CASE WHEN jsonb_typeof(metrics.metrics->'runtime') = 'object'
                   THEN metrics.metrics->'runtime' ELSE '{}'::jsonb END) AS names(key)
               WHERE key !~ $7 OR octet_length(key) > $8
           ) AS invalid_names
    FROM stage_executions stage
    JOIN workflow_runs run USING (run_id)
    LEFT JOIN stage_metrics metrics USING (stage_execution_id)
    WHERE run.owner_id = $1 AND run.run_id = $2
      AND stage.stage_execution_id = $3
), selected AS (
    SELECT owned.*, participant.key, owned.runtime->participant.key AS value
    FROM owned
    LEFT JOIN LATERAL (
        SELECT key
        FROM jsonb_object_keys(CASE WHEN jsonb_typeof(owned.runtime) = 'object'
            THEN owned.runtime ELSE '{}'::jsonb END) AS names(key)
        WHERE key ~ $7 AND octet_length(key) <= $8
          AND key COLLATE "C" > $4 COLLATE "C" ORDER BY key COLLATE "C" LIMIT $5
    ) participant ON true
), projected AS (
    SELECT *, CASE WHEN jsonb_typeof(value->'stopReason') = 'string'
        THEN jsonb_set(value, '{stopReason}', to_jsonb(left(value->>'stopReason', 4096)))
        ELSE value END AS document,
        COALESCE(octet_length(value->>'stopReason') > 4096, false) AS stop_truncated
    FROM selected
)
SELECT terminal, metrics_schema_version, updated_at, tuple_generation,
       jsonb_typeof(runtime), key,
       CASE WHEN octet_length(document::text) <= $6 THEN document END,
       stop_truncated, invalid_names
FROM projected ORDER BY key COLLATE "C" NULLS FIRST
`
