package artifacts

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// DescribeForkRevisions describes the exact target revisions of selected
// bindings' lineage edges of kind, in requested name order. Missing edges are
// omitted; multiple matching edges for one binding are ErrArtifactIntegrity.
// It reads only metadata, never substitutes the current revision, and bounds
// the selection by MaxExactReadBatchSize in the caller's repository/transaction.
func (s ScopedStore) DescribeForkRevisions(ctx context.Context, namespace string, names []string, kind string) ([]contracts.ExactArtifact, error) {
	if err := ValidateForkRevisionSelection(s.scope, namespace, names, kind); err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return []contracts.ExactArtifact{}, nil
	}
	return s.service.repository.DescribeForkRevisions(ctx, s.scope, namespace, names, kind)
}

// DescribeForkSources returns exact source and target identities for selected
// existing bindings, in requested name order. Absent bindings are omitted; an
// existing binding without exactly one matching fork is ErrArtifactIntegrity.
// The request has the same metadata-only batch bound as DescribeForkRevisions.
func (s ScopedStore) DescribeForkSources(ctx context.Context, namespace string, names []string, kind string) ([]LineageEdge, error) {
	if err := ValidateForkRevisionSelection(s.scope, namespace, names, kind); err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return []LineageEdge{}, nil
	}
	return s.service.repository.DescribeForkSources(ctx, s.scope, namespace, names, kind, false)
}

func ValidateForkRevisionSelection(scope Scope, namespace string, names []string, kind string) error {
	if err := ValidateScope(scope); err != nil {
		return err
	}
	if ValidateComponent(namespace) != nil || !validLineageKind(kind) {
		return ErrInvalidName
	}
	if len(names) > MaxExactReadBatchSize {
		return blob.ErrTooLarge
	}
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if ValidateComponent(name) != nil || seen[name] {
			return ErrInvalidName
		}
		seen[name] = true
	}
	return nil
}
