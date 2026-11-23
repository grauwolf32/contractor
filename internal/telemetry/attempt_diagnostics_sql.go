package telemetry

// runtimeDiagnosticPageSQL authorizes one exact Stage through its owning Run,
// then projects only a bounded page of Runtime participants from retained
// StageMetrics. Eval holds retain this authority after the source report rows
// expire. The lateral LIMIT applies before values leave PostgreSQL; neither
// Worker reports nor tool-call arguments are returned. The short SQL stopReason
// prefix is additionally bounded by UTF-8 bytes at the public projection.
// Materialization is intentional: the containing Runtime map is extracted
// once, its keys are enumerated once, and only selected participant values
// enter the next boundary. Carrying runtime through selected rows or inlining
// its extraction repeats large JSONB work for each projected field.
const runtimeDiagnosticPageSQL = `
WITH owned AS MATERIALIZED (
    SELECT stage.terminal_at IS NOT NULL AS terminal,
           metrics.metrics_schema_version, metrics.updated_at,
           metrics.xmin::text AS tuple_generation,
           metrics.metrics->'runtime' AS runtime,
           metrics.metrics->'workers' AS workers
    FROM stage_executions stage
    JOIN workflow_runs run USING (run_id)
    LEFT JOIN stage_metrics metrics USING (stage_execution_id)
    WHERE run.owner_id = $1 AND run.run_id = $2
      AND stage.stage_execution_id = $3
), participant_names AS MATERIALIZED (
    SELECT key FROM owned,
         LATERAL jsonb_object_keys(CASE WHEN jsonb_typeof(owned.runtime) = 'object'
             THEN owned.runtime ELSE '{}'::jsonb END) AS names(key)
), selected AS MATERIALIZED (
    SELECT owned.terminal, owned.metrics_schema_version, owned.updated_at,
           owned.tuple_generation, jsonb_typeof(owned.runtime) AS runtime_kind,
           EXISTS (SELECT 1 FROM participant_names WHERE key !~ $7 OR octet_length(key) > $8) AS invalid_names,
           participant.key, owned.runtime->participant.key AS value,
           owned.workers->participant.key AS worker
    FROM owned
    LEFT JOIN LATERAL (
        SELECT key
        FROM participant_names
        WHERE key ~ $7 AND octet_length(key) <= $8
          AND key COLLATE "C" > $4 COLLATE "C" ORDER BY key COLLATE "C" LIMIT $5
    ) participant ON true
), projected AS MATERIALIZED (
    SELECT terminal, metrics_schema_version, updated_at, tuple_generation,
        runtime_kind, invalid_names, key,
        CASE WHEN worker ? 'compaction' THEN jsonb_build_object('compaction', worker->'compaction')
            ELSE '{}'::jsonb END ||
        CASE WHEN worker ? 'metrics' AND worker->'metrics' ? 'modelPhases'
            THEN jsonb_build_object('modelPhases', worker->'metrics'->'modelPhases') ELSE '{}'::jsonb END AS progress,
        (CASE WHEN jsonb_typeof(value->'stopReason') = 'string'
        THEN jsonb_set(value, '{stopReason}', to_jsonb(left(value->>'stopReason', 4096)))
        ELSE value END)::text AS document,
        COALESCE(octet_length(value->>'stopReason') > 4096, false) AS stop_truncated
    FROM selected
)
SELECT terminal, metrics_schema_version, updated_at, tuple_generation,
       runtime_kind, key,
       CASE WHEN octet_length(document) <= $6 THEN document END,
       stop_truncated, invalid_names,
       CASE WHEN octet_length(progress::text) <= 8192 THEN progress::text END
FROM projected ORDER BY key COLLATE "C" NULLS FIRST
`

// liveWorkerDiagnosticPageSQL selects at most one existing diagnostic page of
// unreleased allocations, through the same exact owner/Run/Stage authority.
// Stage tuple changes fence continuation cursors; no Runtime address is read.
const liveWorkerDiagnosticPageSQL = `
SELECT allocation.allocation_id, allocation.logical_agent_name,
       stage.xmin::text
FROM stage_allocations allocation
JOIN stage_executions stage USING (stage_execution_id)
JOIN workflow_runs run USING (run_id)
WHERE run.owner_id = $1 AND run.run_id = $2
  AND stage.stage_execution_id = $3 AND stage.terminal_at IS NULL
  AND allocation.release_completed_at IS NULL
  AND allocation.logical_agent_name COLLATE "C" > $4 COLLATE "C"
ORDER BY allocation.logical_agent_name COLLATE "C" LIMIT $5
`
