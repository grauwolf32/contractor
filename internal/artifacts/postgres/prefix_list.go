package postgres

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func (r *Repository) ListPrefix(ctx context.Context, scope artifacts.Scope, namespace, prefix string, limit int) ([]artifacts.ArtifactRef, error) {
	// starts_with treats '_' literally, unlike LIKE. Filter and LIMIT execute
	// in PostgreSQL so unrelated names never enter the adapter or HTTP result.
	rows, err := r.db.Query(ctx, `
SELECT namespace, name FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3
  AND starts_with(name, $4)
ORDER BY name
LIMIT $5`, scope.Kind(), scope.ID(), namespace, prefix, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	refs := make([]artifacts.ArtifactRef, 0, limit)
	for rows.Next() {
		var ref artifacts.ArtifactRef
		if err := rows.Scan(&ref.Namespace, &ref.Name); err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}
	return refs, rows.Err()
}
