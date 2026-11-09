package postgres

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/jackc/pgx/v5"
)

func (r *Repository) Metadata(
	ctx context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef,
) (artifacts.Metadata, error) {
	var result artifacts.Metadata
	var revision string
	err := r.db.QueryRow(ctx, metadataSQL,
		scope.Kind(), scope.ID(), ref.Namespace, ref.Name, ref.Revision,
	).Scan(
		&revision, &result.MediaType, &result.Size, &result.Digest, &result.ContentCacheKey, &result.Current,
		&result.Frozen, &result.BindingCreatedAt, &result.CreatedAt, &result.StorageTimestamps.ScopeCreatedAt,
		&result.StorageTimestamps.BlobCreatedAt, &result.StorageTimestamps.VersionCreatedAt,
		&result.StorageTimestamps.BindingUpdatedAt, &result.GitSource,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
	}
	if err != nil {
		return artifacts.Metadata{}, fmt.Errorf("read artifact metadata %s/%s: %w", ref.Namespace, ref.Name, err)
	}
	result.Ref = artifactExactRef(ref.Namespace, ref.Name, revision)
	if err := artifacts.NormalizeMetadataDates(&result); err != nil {
		return artifacts.Metadata{}, err
	}
	return result, nil
}

func (r *Repository) ListMetadata(
	ctx context.Context, scope artifacts.Scope, query artifacts.BindingPageQuery,
) ([]artifacts.Metadata, error) {
	var rows pgx.Rows
	var err error
	if query.MaterialOrder {
		statement := listProjectMaterialMetadataSQL
		if query.Namespace != nil {
			statement = listProjectNamespaceMaterialMetadataSQL
		}
		rows, err = r.db.Query(ctx, statement, scope.ID(), query.Namespace, query.AfterMaterialRank,
			query.AfterNamespace, query.AfterName, query.Limit)
	} else if query.NamePrefix != "" {
		rows, err = r.db.Query(ctx, listProjectPrefixMetadataSQL, scope.ID(), *query.Namespace, query.NamePrefix, query.AfterName, query.Limit)
	} else if scope.Kind() == artifacts.ScopeUser && query.HiddenNamespace == artifactpolicy.AuditStandardCatalogNamespace && query.ExcludeNamespacePrefix == "" && query.HiddenNamespacePrefix == "" {
		rows, err = r.db.Query(ctx, listUserMetadataSQL, scope.ID(), query.Namespace, query.ExcludeNamespace, query.AfterNamespace, query.AfterName, query.Limit, query.HiddenNamespace)
	} else if scope.Kind() == artifacts.ScopeProject &&
		query.ExcludeNamespacePrefix == artifactpolicy.AuditManagedProjectNamespacePrefix &&
		query.HiddenNamespacePrefix == artifactpolicy.HistoricalProjectNamespacePrefix {
		rows, err = r.db.Query(ctx, listProjectMaterialsSQL, scope.ID(), query.Namespace,
			query.ExcludeNamespace, query.AfterNamespace, query.AfterName, query.Limit,
			query.HiddenNamespace)
	} else if scope.Kind() == artifacts.ScopeProject && query.HiddenNamespacePrefix == "" &&
		query.ExcludeNamespacePrefix == artifactpolicy.AuditManagedProjectNamespacePrefix {
		// The generic Project listing skips the Audit-managed namespaces by
		// index; a prefix filter would read every Audit binding (ART-R115).
		rows, err = r.db.Query(ctx, listProjectMetadataSQL, scope.ID(), query.Namespace,
			query.ExcludeNamespace, query.AfterNamespace, query.AfterName, query.Limit,
			query.HiddenNamespace)
	} else {
		rows, err = r.db.Query(ctx, listMetadataSQL, scope.Kind(), scope.ID(), query.Namespace,
			query.ExcludeNamespace, query.ExcludeNamespacePrefix, query.AfterNamespace,
			query.AfterName, query.Limit, query.HiddenNamespace, query.HiddenNamespacePrefix)
	}
	if err != nil {
		return nil, fmt.Errorf("list artifact metadata: %w", err)
	}
	result, err := pgx.AppendRows(make([]artifacts.Metadata, 0, query.Limit), rows, func(row pgx.CollectableRow) (artifacts.Metadata, error) {
		return scanMetadataRow(row, query.MaterialOrder)
	})
	if err != nil {
		return nil, fmt.Errorf("iterate artifact metadata: %w", err)
	}
	return result, nil
}

func (r *Repository) ListVersions(
	ctx context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef, query artifacts.VersionPageQuery,
) ([]artifacts.Metadata, error) {
	rows, err := r.db.Query(ctx, listVersionsSQL, scope.Kind(), scope.ID(), ref.Namespace, ref.Name,
		query.BeforeCreatedAt, query.BeforeRevision, query.Limit)
	if err != nil {
		return nil, fmt.Errorf("list artifact versions: %w", err)
	}
	result, err := pgx.AppendRows(make([]artifacts.Metadata, 0, query.Limit), rows, func(row pgx.CollectableRow) (artifacts.Metadata, error) {
		return scanMetadata(row)
	})
	if err != nil {
		return nil, fmt.Errorf("iterate artifact versions: %w", err)
	}
	return result, nil
}

// ListLineage lists the lineage of exact, an exact revision of scope.
func (r *Repository) ListLineage(
	ctx context.Context, scope artifacts.Scope, exact artifacts.ArtifactRef, query artifacts.LineagePageQuery,
) ([]artifacts.LineageEdge, error) {
	rows, err := r.db.Query(ctx, listLineageSQL, scope.Kind(), scope.ID(), exact.Namespace, exact.Name, *exact.Revision,
		query.ExcludeKind,
		query.BeforeCreatedAt, query.BeforeTargetRevision, query.BeforeSourceRevision,
		query.BeforeKind, query.Limit)
	if err != nil {
		return nil, fmt.Errorf("list artifact lineage: %w", err)
	}
	defer rows.Close()
	result := make([]artifacts.LineageEdge, 0, query.Limit)
	for rows.Next() {
		var edge artifacts.LineageEdge
		var sourceRevision, targetRevision string
		if err := rows.Scan(
			&edge.Kind,
			&edge.SourceScope, &edge.SourceScopeID,
			&edge.Source.Namespace, &edge.Source.Name, &sourceRevision,
			&edge.TargetScope, &edge.TargetScopeID,
			&edge.Target.Namespace, &edge.Target.Name, &targetRevision,
			&edge.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan artifact lineage: %w", err)
		}
		edge.Source.Revision = &sourceRevision
		edge.Target.Revision = &targetRevision
		result = append(result, edge)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate artifact lineage: %w", err)
	}
	return result, nil
}

type metadataRowScanner interface {
	Scan(...any) error
}

func scanMetadata(row metadataRowScanner) (artifacts.Metadata, error) {
	return scanMetadataRow(row, false)
}

func scanMetadataRow(row metadataRowScanner, materialOrder bool) (artifacts.Metadata, error) {
	var result artifacts.Metadata
	var revision string
	destinations := []any{
		&result.Ref.Namespace, &result.Ref.Name, &revision,
		&result.MediaType, &result.Size, &result.Current, &result.Frozen, &result.BindingCreatedAt, &result.CreatedAt, &result.StorageTimestamps.ScopeCreatedAt,
		&result.StorageTimestamps.BlobCreatedAt, &result.StorageTimestamps.VersionCreatedAt,
		&result.StorageTimestamps.BindingUpdatedAt, &result.GitSource,
	}
	if materialOrder {
		destinations = append(destinations, &result.MaterialRank)
	}
	if err := row.Scan(destinations...); err != nil {
		return artifacts.Metadata{}, err
	}
	result.Ref.Revision = &revision
	if err := artifacts.NormalizeMetadataDates(&result); err != nil {
		return artifacts.Metadata{}, err
	}
	return result, nil
}

func artifactExactRef(namespace, name, revision string) artifacts.ArtifactRef {
	value := revision
	return artifacts.ArtifactRef{Namespace: namespace, Name: name, Revision: &value}
}
