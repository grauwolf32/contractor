package evalstore

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// ScopeForMutation also finds retained receipts after experiment purge. It
// authenticates the owner before revealing a Project or replaying a mutation.
func (s *Store) ScopeForMutation(ctx context.Context, owner, id, operation, key string) (Scope, error) {
	var project string
	lookupID := id
	if operation == "submission" || operation == "result" || operation == "assessment" || operation == "checks" {
		separator := strings.LastIndex(id, ":")
		if separator < 1 || evaldomain.Validate("MemberID", bytesOf(id[separator+1:])) != nil {
			return Scope{}, evaldomain.Failure("eval_invalid")
		}
		lookupID = id[:separator]
	}
	err := s.db.QueryRow(ctx, `
SELECT project_id FROM eval_experiments WHERE owner_id=$1 AND experiment_id=$2
UNION SELECT project_id FROM eval_mutation_receipts WHERE owner_id=$1 AND resource_id=$5 AND operation=$3 AND operation_key=$4 LIMIT 1
`, owner, lookupID, operation, key, id).Scan(&project)
	return Scope{OwnerID: owner, ProjectID: project}, normalize(err)
}

func (s *Store) CommandRecord(ctx context.Context, owner, id, command string) (CommandRecord, error) {
	var c CommandRecord
	err := s.db.QueryRow(ctx, `
SELECT c.command_id, c.experiment_id, c.kind, c.state, c.accepted_revision, c.diagnostic, c.created_at, c.finished_at
FROM eval_commands c
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id=$1
    AND e.experiment_id=$2
    AND c.command_id=$3
`, owner, id, command).Scan(&c.ID, &c.ExperimentID, &c.Kind, &c.State, &c.Revision, &c.Diagnostic, &c.CreatedAt, &c.FinishedAt)
	return c, normalize(err)
}

type DatasetPage struct {
	Revision int64
	Items    []evaldomain.Dataset
	HasMore  bool
}

func (s *Store) DatasetPage(ctx context.Context, scope Scope, afterID, afterRevision string, limit int, revision *int64) (DatasetPage, error) {
	if limit < 1 || limit > evaldomain.MaxPageSize {
		return DatasetPage{}, evaldomain.Failure("eval_invalid")
	}
	if _, err := s.project(ctx, scope, false); err != nil {
		return DatasetPage{}, err
	}
	var result DatasetPage
	var raw []byte
	err := s.db.QueryRow(ctx, `
SELECT COALESCE((SELECT revision
        FROM eval_collections
        WHERE owner_id=$1
            AND project_id=$2), 0), COALESCE((SELECT jsonb_agg(metadata
            ORDER BY dataset_id, revision)
        FROM (SELECT dataset_id, revision, metadata
            FROM eval_dataset_revisions
            WHERE owner_id=$1
                AND project_id=$2
                AND (dataset_id, revision)>($3, $4)
            ORDER BY dataset_id, revision LIMIT $5) page), '[]'::jsonb)
`, scope.OwnerID, scope.ProjectID, afterID, afterRevision, limit+1).Scan(&result.Revision, &raw)
	if err != nil {
		return result, err
	}
	if revision != nil && *revision != result.Revision {
		return result, evaldomain.Failure("eval_view_changed")
	}
	if err = json.Unmarshal(raw, &result.Items); err != nil {
		return result, err
	}
	if len(result.Items) > limit {
		result.HasMore = true
		result.Items = result.Items[:limit]
	}
	return result, nil
}

type SummaryPageParams struct {
	ListParams
	AfterTime *time.Time
}

type PublicSummary struct {
	ExperimentSummary
	ExecutionKind string    `json:"executionKind"`
	UpdatedAt     time.Time `json:"updatedAt"`
}

type SummaryPage struct {
	Revision int64
	Items    []PublicSummary
	HasMore  bool
}

func (s *Store) SummaryPage(ctx context.Context, p SummaryPageParams) (SummaryPage, error) {
	if p.OwnerID == "" || p.Limit < 1 || p.Limit > evaldomain.MaxPageSize {
		return SummaryPage{}, evaldomain.Failure("eval_invalid")
	}
	if p.ProjectID != "" {
		if _, err := s.project(ctx, Scope{p.OwnerID, p.ProjectID}, false); err != nil {
			return SummaryPage{}, err
		}
	}
	var result SummaryPage
	var raw []byte
	err := s.db.QueryRow(ctx, `
SELECT COALESCE((SELECT revision FROM eval_collections WHERE owner_id=$1 AND project_id=$2),0),COALESCE((
        SELECT jsonb_agg(jsonb_build_object('experimentId',experiment_id,'projectId',project_id,'name',name,'controlMode',control_mode,'state',state,'revision',revision,'expectedMembers',CASE WHEN expected_count>0 THEN expected_count ELSE jsonb_array_length(convert_from(draft,'UTF8')::jsonb->'caseIds')*(convert_from(draft,'UTF8')::jsonb->>'repetitions')::int*2 END,'executionKind',CASE WHEN draft IS NOT NULL THEN convert_from(draft,'UTF8')::jsonb->'variants'->0->>'kind' ELSE (SELECT convert_from(setup,'UTF8')::jsonb->'variants'->0->>'kind' FROM eval_frozen_plans p WHERE p.experiment_id=page.experiment_id) END,'updatedAt',updated_at) ORDER BY updated_at DESC,experiment_id DESC)
        FROM (SELECT experiment_id,project_id,name,control_mode,state,revision,expected_count,draft,updated_at FROM eval_experiments WHERE owner_id=$1 AND ($2='' OR project_id=$2) AND ($3='' OR state=$3) AND ($4='' OR dataset_id=$4) AND ($5='' OR control_mode=$5) AND ($6::timestamptz IS NULL OR (updated_at,experiment_id)<($6,$7)) ORDER BY updated_at DESC,experiment_id DESC LIMIT $8) page),'[]'::jsonb)
`, p.OwnerID, p.ProjectID, p.State, p.DatasetID, p.ControlMode, p.AfterTime, p.AfterID, p.Limit+1).Scan(&result.Revision, &raw)
	if err != nil {
		return result, err
	}
	if p.Revision != nil && *p.Revision != result.Revision {
		return result, evaldomain.Failure("eval_view_changed")
	}
	if err = json.Unmarshal(raw, &result.Items); err != nil {
		return result, err
	}
	if len(result.Items) > p.Limit {
		result.HasMore = true
		result.Items = result.Items[:p.Limit]
	}
	return result, nil
}
