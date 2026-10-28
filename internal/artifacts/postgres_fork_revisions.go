package artifacts

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

func (r *PostgresRepository) DescribeForkRevisions(ctx context.Context, scope Scope, namespace string, names []string, kind string) ([]contracts.ExactArtifact, error) {
	if err := validateForkRevisionSelection(scope, namespace, names, kind); err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return []contracts.ExactArtifact{}, nil
	}
	rows, err := r.db.Query(ctx, forkTargetRevisionsSQL, scope.kind, scope.id, namespace, names, kind)
	if err != nil {
		return nil, fmt.Errorf("resolve fork target revisions: %w", err)
	}
	type forkTarget struct{ Name, Revision string }
	targets, err := pgx.CollectRows(rows, pgx.RowToStructByPos[forkTarget])
	if err != nil {
		return nil, fmt.Errorf("scan fork target revisions: %w", err)
	}
	seen := make(map[string]bool, len(targets))
	refs := make([]ArtifactRef, 0, len(targets))
	for _, target := range targets {
		if seen[target.Name] {
			return nil, fmt.Errorf("%w: binding %s/%s has multiple %s edges", ErrArtifactIntegrity, namespace, target.Name, kind)
		}
		seen[target.Name] = true
		refs = append(refs, artifactExactRef(namespace, target.Name, target.Revision))
	}
	return r.DescribeExact(ctx, scope, refs)
}
