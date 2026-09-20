package evalstore

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type ListParams struct {
	OwnerID, ProjectID, State, DatasetID, ControlMode, AfterID string
	Limit                                                      int
	Revision                                                   *int64
}

type ExperimentSummary struct {
	ID          string `json:"experimentId"`
	ProjectID   string `json:"projectId"`
	Name        string `json:"name"`
	ControlMode string `json:"controlMode"`
	State       string `json:"state"`
	Revision    int64  `json:"revision"`
	Expected    int    `json:"expectedMembers"`
}

type ExperimentPage struct {
	Revision int64
	Items    []ExperimentSummary
}

// List reads the collection revision and bounded rows from one database snapshot.
// It never selects the draft, dataset document, frozen plan or private checks.
func (s *Store) List(ctx context.Context, p ListParams) (ExperimentPage, error) {
	if p.OwnerID == "" || p.Limit < 1 || p.Limit > evaldomain.MaxPageSize {
		return ExperimentPage{}, evaldomain.Failure("eval_invalid")
	}
	if p.ProjectID != "" {
		if _, err := s.project(ctx, Scope{p.OwnerID, p.ProjectID}, false); err != nil {
			return ExperimentPage{}, err
		}
	}
	var out ExperimentPage
	var raw []byte
	err := s.db.QueryRow(ctx, `
SELECT COALESCE((SELECT revision FROM eval_collections WHERE owner_id=$1 AND project_id=$2),0),COALESCE((
        SELECT jsonb_agg(jsonb_build_object('experimentId',experiment_id,'projectId',project_id,'name',name,'controlMode',control_mode,'state',state,'revision',revision,'expectedMembers',expected_count) ORDER BY experiment_id)
        FROM (SELECT experiment_id,project_id,name,control_mode,state,revision,expected_count FROM eval_experiments
            WHERE owner_id=$1 AND ($2='' OR project_id=$2) AND ($3='' OR state=$3) AND ($4='' OR dataset_id=$4) AND ($5='' OR control_mode=$5) AND experiment_id>$6 ORDER BY experiment_id LIMIT $7) page),'[]'::jsonb)
`, p.OwnerID, p.ProjectID, p.State, p.DatasetID, p.ControlMode, p.AfterID, p.Limit).Scan(&out.Revision, &raw)
	if err != nil {
		return out, err
	}
	if p.Revision != nil && *p.Revision != out.Revision {
		return ExperimentPage{}, evaldomain.Failure("eval_view_changed")
	}
	err = json.Unmarshal(raw, &out.Items)
	return out, err
}
