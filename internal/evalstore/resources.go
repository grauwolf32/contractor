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

// putResources validates every plan resource before it writes, then inserts
// them in freezeBatches by document size. A native plan at the 1,000-member
// ceiling carries about as many case and input documents as members, all
// written inside the freeze transaction that holds the Project and
// experiment locks.
func (s *Store) putResources(ctx context.Context, id string, resources []PlanResource) error {
	if len(resources) > evaldomain.MaxMembers {
		return evaldomain.Failure("eval_limit_exceeded")
	}
	if len(resources) == 0 {
		return nil
	}
	paths := make([]string, 0, len(resources))
	kinds := make([]string, 0, len(resources))
	documents := make([][]byte, 0, len(resources))
	for _, resource := range resources {
		if len(resource.Path) > 1024 || resource.Path == "." || strings.HasPrefix(resource.Path, "/") || strings.Contains(resource.Path, "\\") || path.Clean(resource.Path) != resource.Path || strings.HasPrefix(resource.Path, "../") {
			return evaldomain.Failure("eval_invalid")
		}
		if err := evaldomain.Validate(resource.Document.Kind(), resource.Document.Bytes()); err != nil {
			return err
		}
		paths = append(paths, resource.Path)
		kinds = append(kinds, resource.Document.Kind())
		documents = append(documents, resource.Document.Bytes())
	}
	start := 0
	for _, end := range freezeBatches(len(documents), func(i int) int { return len(documents[i]) }) {
		_, err := s.db.Exec(ctx, `
INSERT INTO eval_plan_resources(experiment_id, resource_path, document_kind, document)
SELECT $1, r.path, r.kind, r.document
  FROM unnest($2::text[], $3::text[], $4::bytea[]) WITH ORDINALITY AS r(path, kind, document, position)
 ORDER BY r.position`, id, paths[start:end], kinds[start:end], documents[start:end])
		if err != nil {
			return normalize(err)
		}
		start = end
	}
	return nil
}

func (s *Store) PlanResource(ctx context.Context, owner, id, resourcePath string) (PlanResource, error) {
	var kind, digest string
	var data []byte
	err := s.db.QueryRow(ctx, `
SELECT r.document_kind, r.document, r.document_sha256
FROM eval_plan_resources r
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id=$1
    AND e.experiment_id=$2
    AND r.resource_path=$3
`, owner, id, resourcePath).Scan(&kind, &data, &digest)
	if err != nil {
		return PlanResource{}, normalize(err)
	}
	// putResources validates before insertion; the immutable row and generated
	// digest have the same trust boundary as FrozenPlan.
	return PlanResource{Path: resourcePath, Document: evaldomain.Restore(kind, data, digest)}, nil
}
