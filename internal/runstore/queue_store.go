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
	rows, err := s.db.Query(ctx, `
WITH page AS (
    SELECT run.run_id, run.project_id, project.name AS project_name,
           project.kind AS project_kind, run.workflow_name,
           run.workflow_version, run.state, run.run_event_generation,
           run.next_run_event_sequence - 1 AS event_sequence,
           run.created_at, run.updated_at
    FROM workflow_runs AS run
    LEFT JOIN projects AS project
      ON project.project_id = run.project_id
     AND project.owner_id = run.owner_id
    WHERE run.owner_id = $1
      AND run.state IN ('initializing', 'running', 'cancelling')
      AND ($2::text IS NULL OR run.state = $2)
      AND (
          $3::text IS NULL
          OR ($3 = 'standalone' AND run.project_id IS NULL)
          OR ($3 = 'project' AND project.kind = 'project')
          OR ($3 = 'evaluation' AND project.kind = 'evaluation')
      )
      AND ($4::timestamptz IS NULL OR (run.created_at, run.run_id) > ($4, $5))
    ORDER BY run.created_at, run.run_id
    LIMIT $6
)
SELECT page.run_id, page.project_id, page.project_name, page.project_kind,
       page.workflow_name, page.workflow_version, page.state,
       page.run_event_generation, page.event_sequence,
       page.created_at, page.updated_at,
       COALESCE(
           jsonb_object_agg(labels.label_key, labels.label_value ORDER BY labels.label_key)
               FILTER (WHERE labels.label_key IS NOT NULL),
           '{}'::jsonb
       )
FROM page
LEFT JOIN workflow_run_metadata_labels AS labels USING (run_id)
GROUP BY page.run_id, page.project_id, page.project_name, page.project_kind,
         page.workflow_name, page.workflow_version, page.state,
         page.run_event_generation, page.event_sequence,
         page.created_at, page.updated_at
ORDER BY page.created_at, page.run_id`,
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
