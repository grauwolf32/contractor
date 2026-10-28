package runstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// DescribeOriginalInputs returns immutable Run-creation input descriptors in
// requested name order, omitting absent inputs. Editable current bindings never
// replace the creation revisions. The shared artifact reader bounds the request
// and uses this store's DB/transaction for metadata-only reads.
func (s *PostgresStore) DescribeOriginalInputs(ctx context.Context, runID string, names []string) ([]contracts.ExactArtifact, error) {
	store, err := artifacts.NewService(artifacts.NewPostgresRepository(s.db)).Run(runID)
	if err != nil {
		return nil, err
	}
	return store.DescribeForkRevisions(ctx, "inputs", names, artifacts.LineageInputFork)
}
