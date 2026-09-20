package evalstore

import (
	"context"
	"path"
	"strings"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type PlanResource struct {
	Path     string
	Document evaldomain.Frozen `json:"-"`
}

func (s *Store) putResources(ctx context.Context, id string, resources []PlanResource) error {
	if len(resources) > evaldomain.MaxMembers {
		return evaldomain.Failure("eval_limit_exceeded")
	}
	for _, resource := range resources {
		if len(resource.Path) > 1024 || resource.Path == "." || strings.HasPrefix(resource.Path, "/") || strings.Contains(resource.Path, "\\") || path.Clean(resource.Path) != resource.Path || strings.HasPrefix(resource.Path, "../") {
			return evaldomain.Failure("eval_invalid")
		}
		if err := evaldomain.Validate(resource.Document.Kind(), resource.Document.Bytes()); err != nil {
			return err
		}
		_, err := s.db.Exec(ctx, `INSERT INTO eval_plan_resources(experiment_id,resource_path,document_kind,document) VALUES($1,$2,$3,$4)`, id, resource.Path, resource.Document.Kind(), resource.Document.Bytes())
		if err != nil {
			return normalize(err)
		}
	}
	return nil
}

func (s *Store) PlanResource(ctx context.Context, owner, id, path string) (PlanResource, error) {
	var kind string
	var data []byte
	err := s.db.QueryRow(ctx, `
SELECT r.document_kind, r.document
FROM eval_plan_resources r
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id=$1
    AND e.experiment_id=$2
    AND r.resource_path=$3
`, owner, id, path).Scan(&kind, &data)
	if err != nil {
		return PlanResource{}, normalize(err)
	}
	document, err := evaldomain.Freeze(kind, data)
	return PlanResource{Path: path, Document: document}, err
}
