package telemetry

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
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
	CollectionPolicy reporting.PerformanceCollectionPolicy `json:"collectionPolicy"`
	Status           AllocationResourceStatus              `json:"status"`
	Reason           *AllocationResourceReason             `json:"reason,omitempty"`
	Resources        *reporting.RuntimeResources           `json:"resources,omitempty"`
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
		var persistedPolicy string
		var releaseCompletedAt *time.Time
		var hasReport bool
		var reportedAllocationID *string
		var encodedResources []byte
		if err := rows.Scan(
			&item.AllocationID, &item.RunID, &item.StageExecutionID,
			&item.Stage, &item.LogicalAgent, &item.Outcome, &item.FinishedAt,
			&persistedPolicy, &releaseCompletedAt, &hasReport,
			&reportedAllocationID, &encodedResources,
		); err != nil {
			return nil, fmt.Errorf("scan allocation resource summary: %w", err)
		}
		item.FinishedAt = item.FinishedAt.UTC()
		item.CollectionPolicy = reporting.PerformanceCollectionPolicy(persistedPolicy)
		if item.CollectionPolicy.ValidatePinned() != nil {
			return nil, errors.New("decode persisted allocation performance collection policy")
		}
		var resources *reporting.RuntimeResources
		if hasReport {
			if reportedAllocationID == nil || *reportedAllocationID != item.AllocationID {
				return nil, errors.New("decode persisted allocation resource report")
			}
			// Full reports are validated on ingestion. Decode only the optional
			// resource block here, preserving RuntimeReport's omission of malformed
			// resources without allocating unrelated Worker detail on every read.
			if len(encodedResources) != 0 {
				if decoded, err := contracts.DecodePrivateStrict[reporting.RuntimeResources](encodedResources); err == nil {
					resources = &decoded
				}
			}
		}
		projectAllocationResources(&item, releaseCompletedAt, hasReport, resources)
		result = append(result, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate allocation resource summaries: %w", err)
	}
	return result, nil
}

func projectAllocationResources(
	item *AllocationResourceSummary,
	releaseCompletedAt *time.Time,
	hasReport bool,
	resources *reporting.RuntimeResources,
) {
	switch item.CollectionPolicy {
	case reporting.PerformanceCollectionDisabled:
		item.Status = AllocationResourceDisabled
		return
	case reporting.PerformanceCollectionUnsupported:
		item.Status = AllocationResourceUnsupported
		return
	}
	if !hasReport {
		if releaseCompletedAt == nil {
			item.Status = AllocationResourcePending
		} else {
			item.Status = AllocationResourceUnavailable
			item.Reason = allocationResourceReason(AllocationResourceReportMissing)
		}
		return
	}
	if resources == nil {
		item.Status = AllocationResourceUnavailable
		item.Reason = allocationResourceReason(AllocationResourceReportMissing)
		return
	}
	if resources.Reason != nil {
		item.Reason = allocationResourceReason(AllocationResourceReason(*resources.Reason))
	}
	switch resources.Status {
	case reporting.ResourceComplete:
		item.Status = AllocationResourceAvailable
	case reporting.ResourcePartial:
		item.Status = AllocationResourcePartial
	default:
		item.Status = AllocationResourceUnavailable
	}
	// invalid_report is a diagnostic sentinel. Do not publish it as a measured
	// resource block because it deliberately contains no observations.
	if resources.Reason == nil || *resources.Reason != reporting.ResourceInvalidReport {
		copy := *resources
		item.Resources = &copy
	}
}

func allocationResourceReason(value AllocationResourceReason) *AllocationResourceReason {
	return &value
}
