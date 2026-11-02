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
	if ValidateComponent(namespace) != nil || ValidateComponent(prefix) != nil ||
		limit < 1 || limit > MaxBindingListLimit {
		return ErrInvalidName
	}
	return nil
}
