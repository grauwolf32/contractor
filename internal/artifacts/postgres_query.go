package artifacts

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/jackc/pgx/v5"
)

func (r *PostgresRepository) Metadata(
	ctx context.Context, scope Scope, ref ArtifactRef,
) (Metadata, error) {
	var result Metadata
	var revision string
	err := r.db.QueryRow(ctx, metadataSQL,
		scope.kind, scope.id, ref.Namespace, ref.Name, ref.Revision,
	).Scan(
		&revision, &result.MediaType, &result.Size, &result.Digest, &result.Current,
		&result.Frozen, &result.CreatedAt, &result.GitSource,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return Metadata{}, ErrArtifactNotFound
	}
	if err != nil {
		return Metadata{}, fmt.Errorf("read artifact metadata %s/%s: %w", ref.Namespace, ref.Name, err)
	}
	result.Ref = artifactExactRef(ref.Namespace, ref.Name, revision)
	return result, nil
}

func (r *PostgresRepository) ListMetadata(
	ctx context.Context, scope Scope, query BindingPageQuery,
) ([]Metadata, error) {
	var rows pgx.Rows
	var err error
	if scope.kind == ScopeProject &&
		query.ExcludeNamespacePrefix == artifactpolicy.AuditManagedProjectNamespacePrefix {
		// The generic Project listing skips the Audit-managed namespaces by
		// index; a prefix filter would read every Audit binding (ART-R115).
		rows, err = r.db.Query(ctx, listProjectMetadataSQL, scope.id, query.Namespace,
			query.ExcludeNamespace, query.AfterNamespace, query.AfterName, query.Limit,
			query.HiddenNamespace)
	} else {
		rows, err = r.db.Query(ctx, listMetadataSQL, scope.kind, scope.id, query.Namespace,
			query.ExcludeNamespace, query.ExcludeNamespacePrefix, query.AfterNamespace,
			query.AfterName, query.Limit, query.HiddenNamespace)
	}
	if err != nil {
		return nil, fmt.Errorf("list artifact metadata: %w", err)
	}
	defer rows.Close()
	result := make([]Metadata, 0, query.Limit)
	for rows.Next() {
		metadata, scanErr := scanMetadata(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan artifact metadata: %w", scanErr)
		}
		result = append(result, metadata)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate artifact metadata: %w", err)
	}
	return result, nil
}

func (r *PostgresRepository) ListVersions(
	ctx context.Context, scope Scope, ref ArtifactRef, query VersionPageQuery,
) ([]Metadata, error) {
	rows, err := r.db.Query(ctx, listVersionsSQL, scope.kind, scope.id, ref.Namespace, ref.Name,
		query.BeforeCreatedAt, query.BeforeRevision, query.Limit)
	if err != nil {
		return nil, fmt.Errorf("list artifact versions: %w", err)
	}
	defer rows.Close()
	result := make([]Metadata, 0, query.Limit)
	for rows.Next() {
		metadata, scanErr := scanMetadata(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan artifact version: %w", scanErr)
		}
		result = append(result, metadata)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate artifact versions: %w", err)
	}
	return result, nil
}

// ListLineage lists the lineage of exact, an exact revision of scope.
func (r *PostgresRepository) ListLineage(
	ctx context.Context, scope Scope, exact ArtifactRef, query LineagePageQuery,
) ([]LineageEdge, error) {
	rows, err := r.db.Query(ctx, listLineageSQL, scope.kind, scope.id, exact.Namespace, exact.Name, *exact.Revision,
		query.ExcludeKind,
		query.BeforeCreatedAt, query.BeforeTargetRevision, query.BeforeSourceRevision,
		query.BeforeKind, query.Limit)
	if err != nil {
		return nil, fmt.Errorf("list artifact lineage: %w", err)
	}
	defer rows.Close()
	result := make([]LineageEdge, 0, query.Limit)
	for rows.Next() {
		var edge LineageEdge
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

func scanMetadata(row metadataRowScanner) (Metadata, error) {
	var result Metadata
	var revision string
	if err := row.Scan(
		&result.Ref.Namespace, &result.Ref.Name, &revision,
		&result.MediaType, &result.Size, &result.Current, &result.Frozen, &result.CreatedAt, &result.GitSource,
	); err != nil {
		return Metadata{}, err
	}
	result.Ref.Revision = &revision
	return result, nil
}

func artifactExactRef(namespace, name, revision string) ArtifactRef {
	value := revision
	return ArtifactRef{Namespace: namespace, Name: name, Revision: &value}
}
