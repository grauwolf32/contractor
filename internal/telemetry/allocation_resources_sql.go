package telemetry

import "fmt"

// SQL statements for allocation_resources.go.

// allocationResourceHistoryQuery selects one of eight stable statement shapes:
// an optional exact Run, inclusive upper bound and exclusive cursor. Omitting
// absent predicates keeps the Run and terminal-time indexes usable even after
// PostgreSQL switches a cached statement to its generic plan. The direct time
// bounds start the index scan at the boundary's tie group; the cross-table row
// comparisons still select the exact allocation within that group.
func allocationResourceHistoryQuery(params AllocationResourceHistoryParams) (string, []any) {
	args := []any{params.OwnerID, params.Limit}
	predicates := ""
	if params.RunID != "" {
		args = append(args, params.RunID)
		predicates += fmt.Sprintf("\n      AND run.run_id = $%d", len(args))
	}
	if params.UpperFinishedAt != nil {
		args = append(args, params.UpperFinishedAt.UTC(), params.UpperAllocationID)
		predicates += fmt.Sprintf(allocationResourceHistoryUpperSQL, len(args)-1, len(args)-1, len(args))
	}
	if params.AfterFinishedAt != nil {
		args = append(args, params.AfterFinishedAt.UTC(), params.AfterAllocationID)
		predicates += fmt.Sprintf(allocationResourceHistoryAfterSQL, len(args)-1, len(args)-1, len(args))
	}
	return fmt.Sprintf(allocationResourceHistorySQL, predicates), args
}

// allocationResourceHistoryUpperSQL adds an index range and the exact inclusive
// upper tuple. Its placeholders are assigned by allocationResourceHistoryQuery.
const allocationResourceHistoryUpperSQL = `
      AND execution.terminal_at <= $%d::timestamptz
      AND (execution.terminal_at, allocation.allocation_id) <= ($%d::timestamptz, $%d::text)`

// allocationResourceHistoryAfterSQL adds an index range and the exact exclusive
// cursor tuple. Its placeholders are assigned by allocationResourceHistoryQuery.
const allocationResourceHistoryAfterSQL = `
      AND execution.terminal_at <= $%d::timestamptz
      AND (execution.terminal_at, allocation.allocation_id) < ($%d::timestamptz, $%d::text)`

// allocationResourceHistorySQL lists owner $1's terminal allocations, newest
// first, up to $2 rows. allocationResourceHistoryQuery fills %s with only the
// enabled bound predicates and appends their values after $2. Each row carries
// whether the allocation ever stored a report, and its best unexpired report
// (complete first, then newest) with resources. Report lookup follows LIMIT.
const allocationResourceHistorySQL = `WITH candidates AS (
    SELECT allocation.allocation_id, execution.run_id, execution.stage_execution_id,
           execution.stage_name, allocation.logical_agent_name, execution.state,
           execution.terminal_at, allocation.performance_collection_policy,
           allocation.release_completed_at, allocation.execution_report_received
    FROM workflow_runs AS run
    JOIN stage_executions AS execution ON execution.run_id = run.run_id
    JOIN stage_allocations AS allocation
      ON allocation.stage_execution_id = execution.stage_execution_id
    WHERE run.owner_id = $1
      AND execution.terminal_at IS NOT NULL%s
    ORDER BY execution.terminal_at DESC, allocation.allocation_id DESC
    LIMIT $2
)
SELECT candidate.allocation_id, candidate.run_id, candidate.stage_execution_id,
       candidate.stage_name, candidate.logical_agent_name, candidate.state,
       candidate.terminal_at, candidate.performance_collection_policy,
       candidate.release_completed_at, candidate.execution_report_received,
       effective.report_id IS NOT NULL,
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
       allocation.execution_report_received,
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
