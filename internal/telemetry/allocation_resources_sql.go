package telemetry

// SQL statements for allocation_resources.go.

// allocationResourceHistorySQL lists owner $1's allocations of terminal
// StageExecutions, newest finished first, up to $10 rows. Boolean flags enable
// the optional filters: $3 for Run $2, $6 for the inclusive upper bound
// ($4, $5) and $9 for the exclusive cursor ($7, $8). Each row carries the best
// unexpired execution report (complete first, then newest) and its resources.
// Used by Repository.ListAllocationResourceHistory.
const allocationResourceHistorySQL = `WITH candidates AS (
    SELECT allocation.allocation_id, execution.run_id, execution.stage_execution_id,
           execution.stage_name, allocation.logical_agent_name, execution.state,
           execution.terminal_at, allocation.performance_collection_policy,
           allocation.release_completed_at
    FROM workflow_runs AS run
    JOIN stage_executions AS execution ON execution.run_id = run.run_id
    JOIN stage_allocations AS allocation
      ON allocation.stage_execution_id = execution.stage_execution_id
    WHERE run.owner_id = $1
      AND execution.terminal_at IS NOT NULL
      AND (NOT $3::boolean OR run.run_id = $2)
      AND (NOT $6::boolean OR
           (execution.terminal_at, allocation.allocation_id) <= ($4::timestamptz, $5::text))
      AND (NOT $9::boolean OR
           (execution.terminal_at, allocation.allocation_id) < ($7::timestamptz, $8::text))
    ORDER BY execution.terminal_at DESC, allocation.allocation_id DESC
    LIMIT $10
)
SELECT candidate.allocation_id, candidate.run_id, candidate.stage_execution_id,
       candidate.stage_name, candidate.logical_agent_name, candidate.state,
       candidate.terminal_at, candidate.performance_collection_policy,
       candidate.release_completed_at, effective.report_id IS NOT NULL,
       effective.reported_allocation_id, effective.resources
FROM candidates AS candidate
LEFT JOIN LATERAL (
    SELECT report.report_id,
           CASE WHEN jsonb_typeof(report.report->'allocationId') = 'string'
                THEN report.report->>'allocationId' END AS reported_allocation_id,
           report.report->'runtime'->'resources' AS resources
    FROM allocation_execution_reports AS report
    WHERE report.allocation_id = candidate.allocation_id
      AND report.expires_at > statement_timestamp()
    ORDER BY COALESCE(
                 (report.report->'worker'->>'complete')::boolean
                 AND (report.report->'runtime'->>'complete')::boolean,
                 false
             ) DESC,
             report.received_at DESC, report.report_id
    LIMIT 1
) AS effective ON true
ORDER BY candidate.terminal_at DESC, candidate.allocation_id DESC`

// stageAllocationResourcesSQL returns every allocation of owner $1's terminal
// StageExecutions listed in $2, with the same effective report and runtime
// resources as allocationResourceHistorySQL, ordered by StageExecution and
// logical agent. Used by Repository.ListStageAllocationResources.
const stageAllocationResourcesSQL = `SELECT allocation.allocation_id, execution.run_id,
       execution.stage_execution_id, execution.stage_name,
       allocation.logical_agent_name, execution.state, execution.terminal_at,
       allocation.performance_collection_policy, allocation.release_completed_at,
       effective.report_id IS NOT NULL, effective.reported_allocation_id,
       effective.resources
FROM workflow_runs AS run
JOIN stage_executions AS execution ON execution.run_id = run.run_id
JOIN stage_allocations AS allocation
  ON allocation.stage_execution_id = execution.stage_execution_id
LEFT JOIN LATERAL (
    SELECT report.report_id,
           CASE WHEN jsonb_typeof(report.report->'allocationId') = 'string'
                THEN report.report->>'allocationId' END AS reported_allocation_id,
           report.report->'runtime'->'resources' AS resources
    FROM allocation_execution_reports AS report
    WHERE report.allocation_id = allocation.allocation_id
      AND report.expires_at > statement_timestamp()
    ORDER BY COALESCE(
                 (report.report->'worker'->>'complete')::boolean
                 AND (report.report->'runtime'->>'complete')::boolean,
                 false
             ) DESC,
             report.received_at DESC, report.report_id
    LIMIT 1
) AS effective ON true
WHERE run.owner_id = $1
  AND execution.stage_execution_id = ANY($2::text[])
  AND execution.terminal_at IS NOT NULL
ORDER BY execution.stage_execution_id, allocation.logical_agent_name`
