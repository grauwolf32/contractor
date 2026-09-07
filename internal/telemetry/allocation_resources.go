package telemetry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

type AllocationResourceStatus string
type AllocationResourceReason string

const (
	AllocationResourceDisabled    AllocationResourceStatus = "disabled"
	AllocationResourceUnsupported AllocationResourceStatus = "unsupported"
	AllocationResourcePending     AllocationResourceStatus = "pending"
	AllocationResourceAvailable   AllocationResourceStatus = "available"
	AllocationResourcePartial     AllocationResourceStatus = "partial"
	AllocationResourceUnavailable AllocationResourceStatus = "unavailable"

	AllocationResourceLegacy        AllocationResourceReason = "legacy"
	AllocationResourceReportMissing AllocationResourceReason = "report_missing"
)

type AllocationResourceSummary struct {
	AllocationID     string                                `json:"allocationId"`
	RunID            string                                `json:"runId"`
	StageExecutionID string                                `json:"stageExecutionId"`
	Stage            string                                `json:"stage"`
	LogicalAgent     string                                `json:"logicalAgent"`
	Outcome          string                                `json:"outcome"`
	FinishedAt       time.Time                             `json:"finishedAt"`
	CollectionPolicy contracts.PerformanceCollectionPolicy `json:"collectionPolicy"`
	Status           AllocationResourceStatus              `json:"status"`
	Reason           *AllocationResourceReason             `json:"reason,omitempty"`
	Resources        *contracts.RuntimeResources           `json:"resources,omitempty"`
}

type AllocationResourceHistoryParams struct {
	OwnerID           string
	RunID             string
	UpperFinishedAt   *time.Time
	UpperAllocationID string
	AfterFinishedAt   *time.Time
	AfterAllocationID string
	Limit             int
}

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
       candidate.release_completed_at, effective.report
FROM candidates AS candidate
LEFT JOIN LATERAL (
    SELECT report.report
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

func (r *Repository) ListAllocationResourceHistory(
	ctx context.Context,
	params AllocationResourceHistoryParams,
) ([]AllocationResourceSummary, error) {
	if err := validateAllocationResourceHistoryParams(params); err != nil {
		return nil, err
	}
	upperAt, hasUpper := time.Time{}, params.UpperFinishedAt != nil
	if hasUpper {
		upperAt = params.UpperFinishedAt.UTC()
	}
	afterAt, hasAfter := time.Time{}, params.AfterFinishedAt != nil
	if hasAfter {
		afterAt = params.AfterFinishedAt.UTC()
	}
	rows, err := r.db.Query(ctx, allocationResourceHistorySQL,
		params.OwnerID, params.RunID, params.RunID != "",
		upperAt, params.UpperAllocationID, hasUpper,
		afterAt, params.AfterAllocationID, hasAfter, params.Limit,
	)
	if err != nil {
		return nil, fmt.Errorf("list allocation resource history: %w", err)
	}
	defer rows.Close()
	return scanAllocationResourceSummaries(rows)
}

const stageAllocationResourcesSQL = `SELECT allocation.allocation_id, execution.run_id,
       execution.stage_execution_id, execution.stage_name,
       allocation.logical_agent_name, execution.state, execution.terminal_at,
       allocation.performance_collection_policy, allocation.release_completed_at,
       effective.report
FROM workflow_runs AS run
JOIN stage_executions AS execution ON execution.run_id = run.run_id
JOIN stage_allocations AS allocation
  ON allocation.stage_execution_id = execution.stage_execution_id
LEFT JOIN LATERAL (
    SELECT report.report
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

func (r *Repository) ListStageAllocationResources(
	ctx context.Context,
	ownerID string,
	stageExecutionIDs []string,
) (map[string][]AllocationResourceSummary, error) {
	if err := requireText("ownerID", ownerID); err != nil {
		return nil, err
	}
	if len(stageExecutionIDs) == 0 {
		return map[string][]AllocationResourceSummary{}, nil
	}
	if len(stageExecutionIDs) > 1024 {
		return nil, fmt.Errorf("%w: too many StageExecution IDs", ErrInvalid)
	}
	for _, id := range stageExecutionIDs {
		if err := requireText("stageExecutionID", id); err != nil {
			return nil, err
		}
	}
	rows, err := r.db.Query(ctx, stageAllocationResourcesSQL, ownerID, stageExecutionIDs)
	if err != nil {
		return nil, fmt.Errorf("list Stage allocation resources: %w", err)
	}
	defer rows.Close()
	items, err := scanAllocationResourceSummaries(rows)
	if err != nil {
		return nil, err
	}
	result := make(map[string][]AllocationResourceSummary)
	for _, item := range items {
		result[item.StageExecutionID] = append(result[item.StageExecutionID], item)
	}
	return result, nil
}

func validateAllocationResourceHistoryParams(params AllocationResourceHistoryParams) error {
	if err := requireText("ownerID", params.OwnerID); err != nil {
		return err
	}
	if params.RunID != "" {
		if err := requireText("runID", params.RunID); err != nil {
			return err
		}
	}
	if params.Limit < 1 || params.Limit > 101 ||
		(params.UpperFinishedAt == nil) != (params.UpperAllocationID == "") ||
		(params.AfterFinishedAt == nil) != (params.AfterAllocationID == "") {
		return fmt.Errorf("%w: invalid allocation history bounds", ErrInvalid)
	}
	if params.UpperFinishedAt != nil && (params.UpperFinishedAt.IsZero() || params.AfterFinishedAt != nil && params.AfterFinishedAt.After(*params.UpperFinishedAt)) {
		return fmt.Errorf("%w: invalid allocation history time bounds", ErrInvalid)
	}
	return nil
}

func scanAllocationResourceSummaries(rows pgx.Rows) ([]AllocationResourceSummary, error) {
	result := make([]AllocationResourceSummary, 0)
	for rows.Next() {
		var item AllocationResourceSummary
		var persistedPolicy *string
		var releaseCompletedAt *time.Time
		var encodedReport []byte
		if err := rows.Scan(
			&item.AllocationID, &item.RunID, &item.StageExecutionID,
			&item.Stage, &item.LogicalAgent, &item.Outcome, &item.FinishedAt,
			&persistedPolicy, &releaseCompletedAt, &encodedReport,
		); err != nil {
			return nil, fmt.Errorf("scan allocation resource summary: %w", err)
		}
		item.FinishedAt = item.FinishedAt.UTC()
		var report *contracts.AllocationFinalReport
		if len(encodedReport) != 0 {
			var decoded contracts.AllocationFinalReport
			if err := json.Unmarshal(encodedReport, &decoded); err != nil || decoded.Validate() != nil || decoded.AllocationID != item.AllocationID {
				return nil, errors.New("decode persisted allocation resource report")
			}
			report = &decoded
		}
		projectAllocationResources(&item, persistedPolicy, releaseCompletedAt, report)
		result = append(result, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate allocation resource summaries: %w", err)
	}
	return result, nil
}

func projectAllocationResources(
	item *AllocationResourceSummary,
	persistedPolicy *string,
	releaseCompletedAt *time.Time,
	report *contracts.AllocationFinalReport,
) {
	if persistedPolicy == nil {
		item.CollectionPolicy = contracts.PerformanceCollectionLegacy
		item.Status = AllocationResourceUnavailable
		item.Reason = allocationResourceReason(AllocationResourceLegacy)
		return
	}
	item.CollectionPolicy = contracts.PerformanceCollectionPolicy(*persistedPolicy)
	switch item.CollectionPolicy {
	case contracts.PerformanceCollectionDisabled:
		item.Status = AllocationResourceDisabled
		return
	case contracts.PerformanceCollectionUnsupported:
		item.Status = AllocationResourceUnsupported
		return
	case contracts.PerformanceCollectionRequested:
	default:
		item.Status = AllocationResourceUnavailable
		item.Reason = allocationResourceReason(AllocationResourceLegacy)
		item.CollectionPolicy = contracts.PerformanceCollectionLegacy
		return
	}
	if report == nil {
		if releaseCompletedAt == nil {
			item.Status = AllocationResourcePending
		} else {
			item.Status = AllocationResourceUnavailable
			item.Reason = allocationResourceReason(AllocationResourceReportMissing)
		}
		return
	}
	resources := report.Runtime.Resources
	if resources == nil {
		item.Status = AllocationResourceUnavailable
		item.Reason = allocationResourceReason(AllocationResourceReportMissing)
		return
	}
	if resources.Reason != nil {
		item.Reason = allocationResourceReason(AllocationResourceReason(*resources.Reason))
	}
	switch resources.Status {
	case contracts.ResourceComplete:
		item.Status = AllocationResourceAvailable
	case contracts.ResourcePartial:
		item.Status = AllocationResourcePartial
	default:
		item.Status = AllocationResourceUnavailable
	}
	// invalid_report is a diagnostic sentinel. Do not publish it as a measured
	// resource block because it deliberately contains no observations.
	if resources.Reason == nil || *resources.Reason != contracts.ResourceInvalidReport {
		copy := *resources
		item.Resources = &copy
	}
}

func allocationResourceReason(value AllocationResourceReason) *AllocationResourceReason {
	return &value
}
