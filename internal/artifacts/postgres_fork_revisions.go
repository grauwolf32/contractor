package artifacts

import (
	"context"
	"fmt"
	"time"

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

func (r *PostgresRepository) DescribeForkSources(ctx context.Context, scope Scope, namespace string, names []string, kind string) ([]LineageEdge, error) {
	if err := validateForkRevisionSelection(scope, namespace, names, kind); err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return []LineageEdge{}, nil
	}
	rows, err := r.db.Query(ctx, forkSourceRevisionsSQL, scope.kind, scope.id, namespace, names, kind)
	if err != nil {
		return nil, fmt.Errorf("resolve fork source revisions: %w", err)
	}
	defer rows.Close()
	result := make([]LineageEdge, 0, len(names))
	seen := make(map[string]bool, len(names))
	for rows.Next() {
		var name string
		var targetRevision, sourceKind, sourceID, sourceNamespace, sourceName, sourceRevision *string
		var createdAt *time.Time
		if err := rows.Scan(&name, &targetRevision, &sourceKind, &sourceID, &sourceNamespace, &sourceName, &sourceRevision, &createdAt); err != nil {
			return nil, fmt.Errorf("scan fork source revision: %w", err)
		}
		if seen[name] || targetRevision == nil || sourceKind == nil || sourceID == nil || sourceNamespace == nil || sourceName == nil || sourceRevision == nil || createdAt == nil {
			return nil, fmt.Errorf("%w: binding %s/%s requires exactly one %s fork", ErrArtifactIntegrity, namespace, name, kind)
		}
		seen[name] = true
		result = append(result, LineageEdge{
			TargetScope: scope.kind, TargetScopeID: scope.id,
			Target:      artifactExactRef(namespace, name, *targetRevision),
			SourceScope: ScopeKind(*sourceKind), SourceScopeID: *sourceID,
			Source: artifactExactRef(*sourceNamespace, *sourceName, *sourceRevision),
			Kind:   kind, CreatedAt: *createdAt,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read fork source revisions: %w", err)
	}
	return result, nil
}
