package runstore

import (
	"context"
	"encoding/json"
	"fmt"
)

// ListRunQueue returns only non-terminal Runs in deterministic oldest-first
// creation order. This read model is deliberately independent from Scheduler
// claim priority and never locks or mutates a Run.
func (s *PostgresStore) ListRunQueue(
	ctx context.Context,
	params ListRunQueueParams,
) ([]WorkflowRunQueueItem, error) {
	if err := validateRunQueueParams(params); err != nil {
		return nil, err
	}
	var state *string
	if params.State != nil {
		value := string(*params.State)
		state = &value
	}
	var membership *string
	if params.Membership != nil {
		value := string(*params.Membership)
		membership = &value
	}
	rows, err := s.db.Query(ctx, listRunQueueSQL,
		params.OwnerID, state, membership, params.AfterCreatedAt,
		params.AfterRunID, params.Limit,
	)
	if err != nil {
		return nil, fmt.Errorf("list WorkflowRun queue for owner: %w", err)
	}
	defer rows.Close()
	result := make([]WorkflowRunQueueItem, 0, params.Limit)
	for rows.Next() {
		var item WorkflowRunQueueItem
		var projectName, projectKind *string
		var state string
		var encodedLabels []byte
		if err := rows.Scan(
			&item.RunID, &item.ProjectID, &projectName, &projectKind,
			&item.WorkflowName, &item.WorkflowVersion, &state,
			&item.EventCursor.Generation, &item.EventCursor.Sequence,
			&item.CreatedAt, &item.UpdatedAt, &encodedLabels,
		); err != nil {
			return nil, fmt.Errorf("scan WorkflowRun queue page: %w", err)
		}
		item.State = WorkflowRunState(state)
		if item.ProjectID == nil {
			if projectName != nil || projectKind != nil {
				return nil, fmt.Errorf("stored standalone queue item has Project metadata")
			}
		} else {
			if projectName == nil || projectKind == nil ||
				*projectName == "" || (*projectKind != "project" && *projectKind != "evaluation") {
				return nil, fmt.Errorf("stored Project queue item is invalid")
			}
			item.ProjectName = *projectName
			item.ProjectKind = *projectKind
		}
		if item.EventCursor.Sequence < 0 || item.EventCursor.Generation == "" {
			return nil, fmt.Errorf("stored WorkflowRun queue event cursor is invalid")
		}
		if err := json.Unmarshal(encodedLabels, &item.MetadataLabels); err != nil {
			return nil, fmt.Errorf("decode WorkflowRun queue metadata labels: %w", err)
		}
		if err := item.MetadataLabels.Validate(); err != nil {
			return nil, fmt.Errorf("validate WorkflowRun queue metadata labels: %w", err)
		}
		result = append(result, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate WorkflowRun queue page: %w", err)
	}
	return result, nil
}

func validateRunQueueParams(params ListRunQueueParams) error {
	if err := validateOpaque("ownerID", params.OwnerID); err != nil {
		return err
	}
	if params.State != nil &&
		*params.State != RunInitializing &&
		*params.State != RunRunning &&
		*params.State != RunPending &&
		*params.State != RunWaiting &&
		*params.State != RunCancelling {
		return invalidf("WorkflowRun queue state must be non-terminal")
	}
	if params.Membership != nil && !params.Membership.Valid() {
		return invalidf("WorkflowRun queue membership is invalid")
	}
	if params.Limit < 1 || params.Limit > 201 {
		return invalidf("WorkflowRun queue page limit must be between 1 and 201")
	}
	if (params.AfterCreatedAt == nil) != (params.AfterRunID == "") {
		return invalidf("WorkflowRun queue keyset is incomplete")
	}
	if params.AfterCreatedAt != nil {
		if params.AfterCreatedAt.IsZero() {
			return invalidf("WorkflowRun queue timestamp is invalid")
		}
		if err := validateOpaque("afterRunID", params.AfterRunID); err != nil {
			return err
		}
	}
	return nil
}
