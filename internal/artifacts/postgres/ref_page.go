package postgres

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
)

func (r *Repository) ListRefs(ctx context.Context, scope artifacts.Scope, query artifacts.RefPageQuery) (artifacts.RefPage, error) {
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
	rows, err := r.db.Query(ctx, statement, scope.Kind(), scope.ID(), start, query.AfterName,
		hiddenNamespaces, exemptNamespaces, hiddenPrefixes, query.Limit+1)
	if err != nil {
		return artifacts.RefPage{}, fmt.Errorf("list artifact refs: %w", err)
	}
	defer rows.Close()
	refs := make([]artifacts.ArtifactRef, 0, query.Limit+1)
	for rows.Next() {
		var ref artifacts.ArtifactRef
		if err := rows.Scan(&ref.Namespace, &ref.Name); err != nil {
			return artifacts.RefPage{}, fmt.Errorf("scan artifact ref: %w", err)
		}
		// The SQL exclusion is this predicate; disagreement is a defect, and
		// a hidden binding must never reach a model-facing page.
		if query.ModelVisible && artifactpolicy.IsModelHiddenBinding(ref.Namespace, ref.Name) {
			return artifacts.RefPage{}, fmt.Errorf("%w: model-visible listing returned a hidden binding", blob.ErrIntegrity)
		}
		refs = append(refs, ref)
	}
	if err := rows.Err(); err != nil {
		return artifacts.RefPage{}, fmt.Errorf("iterate artifact refs: %w", err)
	}
	if len(refs) > query.Limit {
		return artifacts.RefPage{Refs: refs[:query.Limit], More: true}, nil
	}
	return artifacts.RefPage{Refs: refs}, nil
}
