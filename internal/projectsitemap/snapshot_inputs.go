package projectsitemap

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

// resolveSnapshotInputs reads only the bounded configured original input forks,
// inside the snapshot transaction. Missing optional inputs cannot block a map
// read; selectSnapshot records a gap only when the selection needs their bytes.
func resolveSnapshotInputs(ctx context.Context, tx pgx.Tx, access AllocationAccess) ([]contracts.ExactArtifact, error) {
	if access.Policy.Validate() != nil {
		return nil, ErrInvalid
	}
	if len(access.Policy.SnapshotInputs) == 0 {
		return []contracts.ExactArtifact{}, nil
	}
	rows, err := tx.Query(ctx, snapshotInputRevisionsSQL, access.RunID, access.Policy.SnapshotInputs)
	if err != nil {
		return nil, err
	}
	type originalInput struct{ Name, Revision string }
	originals, err := pgx.CollectRows(rows, pgx.RowToStructByPos[originalInput])
	if err != nil {
		return nil, err
	}
	seen := make(map[string]bool, len(originals))
	refs := make([]contracts.ArtifactRef, 0, len(originals))
	for _, original := range originals {
		if seen[original.Name] {
			return nil, ErrInvalid
		}
		seen[original.Name] = true
		refs = append(refs, contracts.ArtifactRef{Namespace: "inputs", Name: original.Name, Revision: &original.Revision})
	}
	if len(refs) == 0 {
		return []contracts.ExactArtifact{}, nil
	}
	store, err := artifacts.NewService(artifacts.NewPostgresRepository(tx)).Run(access.RunID)
	if err != nil {
		return nil, err
	}
	return store.DescribeExact(ctx, refs)
}
