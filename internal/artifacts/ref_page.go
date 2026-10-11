package artifacts

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
)

// ListRefs returns one keyset page of at most query.Limit versionless current
// bindings in namespace and name order. A page never holds a binding the query
// excludes, and every binding present for the whole traversal is returned
// exactly once across the pages.
func (s ScopedStore) ListRefs(ctx context.Context, query RefPageQuery) (RefPage, error) {
	if err := validateScope(s.scope); err != nil {
		return RefPage{}, err
	}
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
		if err := validateComponent(*query.Namespace); err != nil {
			return err
		}
	}
	if (query.AfterNamespace == "") != (query.AfterName == "") {
		return ErrInvalidName
	}
	if query.AfterNamespace == "" {
		return nil
	}
	if validateComponent(query.AfterNamespace) != nil || validateComponent(query.AfterName) != nil {
		return ErrInvalidName
	}
	// A position from another Namespace does not belong to this traversal.
	if query.Namespace != nil && query.AfterNamespace != *query.Namespace {
		return ErrInvalidName
	}
	return nil
}

func (r *PostgresRepository) ListRefs(ctx context.Context, scope Scope, query RefPageQuery) (RefPage, error) {
	if err := validateScope(scope); err != nil {
		return RefPage{}, err
	}
	if err := validateRefPageQuery(query); err != nil {
		return RefPage{}, err
	}
	hiddenNamespaces, exemptNamespaces, hiddenPrefixes := []string{}, []string{}, []string{}
	if query.ModelVisible {
		hiddenNamespaces = artifactpolicy.ModelHiddenNamespaces()
		exemptNamespaces = artifactpolicy.ModelNamePrefixExemptNamespaces()
		hiddenPrefixes = artifactpolicy.ModelHiddenNamePrefixes()
	}
	// One extra row reports whether another page follows. Each statement is a
	// range scan of the binding primary key that starts at the keyset position.
	statement, start := listRefPageSQL, query.AfterNamespace
	if query.Namespace != nil {
		statement, start = listNamespaceRefPageSQL, *query.Namespace
	}
	rows, err := r.db.Query(ctx, statement, scope.kind, scope.id, start, query.AfterName,
		hiddenNamespaces, exemptNamespaces, hiddenPrefixes, query.Limit+1)
	if err != nil {
		return RefPage{}, fmt.Errorf("list artifact refs: %w", err)
	}
	defer rows.Close()
	refs := make([]ArtifactRef, 0, query.Limit+1)
	for rows.Next() {
		var ref ArtifactRef
		if err := rows.Scan(&ref.Namespace, &ref.Name); err != nil {
			return RefPage{}, fmt.Errorf("scan artifact ref: %w", err)
		}
		// The SQL exclusion is this predicate; disagreement is a defect, and
		// a hidden binding must never reach a model-facing page.
		if query.ModelVisible && artifactpolicy.IsModelHiddenBinding(ref.Namespace, ref.Name) {
			return RefPage{}, fmt.Errorf("%w: model-visible listing returned a hidden binding", ErrArtifactIntegrity)
		}
		refs = append(refs, ref)
	}
	if err := rows.Err(); err != nil {
		return RefPage{}, fmt.Errorf("iterate artifact refs: %w", err)
	}
	if len(refs) > query.Limit {
		return RefPage{Refs: refs[:query.Limit], More: true}, nil
	}
	return RefPage{Refs: refs}, nil
}
