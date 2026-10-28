package artifacts

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

func (r *PostgresRepository) DescribeOriginalInputs(ctx context.Context, scope Scope, names []string) ([]contracts.ExactArtifact, error) {
	if err := validateOriginalInputs(scope, names); err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return []contracts.ExactArtifact{}, nil
	}
	rows, err := r.db.Query(ctx, originalInputRevisionsSQL, scope.id, names)
	if err != nil {
		return nil, fmt.Errorf("resolve original Run inputs: %w", err)
	}
	type originalInput struct{ Name, Revision string }
	originals, err := pgx.CollectRows(rows, pgx.RowToStructByPos[originalInput])
	if err != nil {
		return nil, fmt.Errorf("scan original Run inputs: %w", err)
	}
	seen := make(map[string]bool, len(originals))
	refs := make([]ArtifactRef, 0, len(originals))
	for _, original := range originals {
		if seen[original.Name] {
			return nil, fmt.Errorf("%w: input %s has multiple original forks", ErrArtifactIntegrity, original.Name)
		}
		seen[original.Name] = true
		refs = append(refs, artifactExactRef("inputs", original.Name, original.Revision))
	}
	return r.DescribeExact(ctx, scope, refs)
}
