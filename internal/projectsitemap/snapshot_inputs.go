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
	store, err := artifacts.NewService(artifacts.NewPostgresRepository(tx)).Run(access.RunID)
	if err != nil {
		return nil, err
	}
	return store.DescribeOriginalInputs(ctx, access.Policy.SnapshotInputs)
}
