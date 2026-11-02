package artifacts

import (
	"context"
)

// ListRefs returns one keyset page of at most query.Limit versionless current
// bindings in namespace and name order. A page never holds a binding the query
// excludes, and every binding present for the whole traversal is returned
// exactly once across the pages.
func (s ScopedStore) ListRefs(ctx context.Context, query RefPageQuery) (RefPage, error) {
	if err := validateRefPageQuery(query); err != nil {
		return RefPage{}, err
	}
	return s.service.repository.ListRefs(ctx, s.scope, query)
}

func validateRefPageQuery(query RefPageQuery) error {
	if query.Limit < 1 || query.Limit > MaxBindingListLimit {
		return ErrInvalidName
	}
	if query.Namespace != nil {
		if err := ValidateComponent(*query.Namespace); err != nil {
			return err
		}
	}
	if (query.AfterNamespace == "") != (query.AfterName == "") {
		return ErrInvalidName
	}
	if query.AfterNamespace == "" {
		return nil
	}
	if ValidateComponent(query.AfterNamespace) != nil || ValidateComponent(query.AfterName) != nil {
		return ErrInvalidName
	}
	// A position from another Namespace does not belong to this traversal.
	if query.Namespace != nil && query.AfterNamespace != *query.Namespace {
		return ErrInvalidName
	}
	return nil
}
