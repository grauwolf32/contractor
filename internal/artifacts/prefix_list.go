package artifacts

import (
	"context"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// MaxBindingListLimit bounds every binding listing (ListRefs, ListPrefix)
// independently of the number of bindings in the scope (LIM-ART-3). The
// Runtime receives this policy through CTL-R171 shared configuration.
const MaxBindingListLimit = contracts.MaxArtifactListItems

// ListPrefix returns at most limit versionless refs in name order. Callers
// checking a quota must request one extra result to detect overflow.
func (s ScopedStore) ListPrefix(ctx context.Context, namespace, prefix string, limit int) ([]ArtifactRef, error) {
	if err := validatePrefixList(namespace, prefix, limit); err != nil {
		return nil, err
	}
	return s.service.repository.ListPrefix(ctx, s.scope, namespace, prefix, limit)
}

func validatePrefixList(namespace, prefix string, limit int) error {
	if validateComponent(namespace) != nil || validateComponent(prefix) != nil ||
		limit < 1 || limit > MaxBindingListLimit {
		return ErrInvalidName
	}
	return nil
}

func (r *PostgresRepository) ListPrefix(ctx context.Context, scope Scope, namespace, prefix string, limit int) ([]ArtifactRef, error) {
	// starts_with treats '_' literally, unlike LIKE. Filter and LIMIT execute
	// in PostgreSQL so unrelated names never enter the adapter or HTTP result.
	rows, err := r.db.Query(ctx, `
SELECT namespace, name FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3
  AND starts_with(name, $4)
ORDER BY name
LIMIT $5`, scope.kind, scope.id, namespace, prefix, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	refs := make([]ArtifactRef, 0, limit)
	for rows.Next() {
		var ref ArtifactRef
		if err := rows.Scan(&ref.Namespace, &ref.Name); err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}
	return refs, rows.Err()
}
