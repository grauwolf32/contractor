package artifacts

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

func (r *PostgresRepository) Metadata(
	ctx context.Context, scope Scope, ref ArtifactRef,
) (Metadata, error) {
	if err := validateScope(scope); err != nil {
		return Metadata{}, err
	}
	if err := validateRef(ref); err != nil {
		return Metadata{}, err
	}
	var result Metadata
	var revision string
	err := r.db.QueryRow(ctx, `
SELECT revision.revision, version.media_type, blob.size_bytes,
       'sha256:' || encode(blob.sha256, 'hex'),
       revision.revision = binding.current_revision, binding.frozen, revision.created_at
FROM artifact_bindings AS binding
JOIN artifact_binding_revisions AS revision
  ON revision.scope_kind = binding.scope_kind
 AND revision.scope_id = binding.scope_id
 AND revision.namespace = binding.namespace
 AND revision.name = binding.name
JOIN artifact_versions AS version ON version.version_id = revision.version_id
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE binding.scope_kind = $1 AND binding.scope_id = $2
  AND binding.namespace = $3 AND binding.name = $4
  AND (($5::text IS NULL AND revision.revision = binding.current_revision)
       OR ($5::text IS NOT NULL AND revision.revision = $5))`,
		scope.kind, scope.id, ref.Namespace, ref.Name, ref.Revision,
	).Scan(
		&revision, &result.MediaType, &result.Size, &result.Digest, &result.Current,
		&result.Frozen, &result.CreatedAt,
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
	if err := validateScope(scope); err != nil {
		return nil, err
	}
	if err := validateBindingPageQuery(query); err != nil {
		return nil, err
	}
	rows, err := r.db.Query(ctx, `
SELECT binding.namespace, binding.name, revision.revision,
       version.media_type, blob.size_bytes, true, binding.frozen, revision.created_at
FROM artifact_bindings AS binding
JOIN artifact_binding_revisions AS revision
  ON revision.scope_kind = binding.scope_kind
 AND revision.scope_id = binding.scope_id
 AND revision.namespace = binding.namespace
 AND revision.name = binding.name
 AND revision.revision = binding.current_revision
JOIN artifact_versions AS version ON version.version_id = revision.version_id
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE binding.scope_kind = $1 AND binding.scope_id = $2
  AND ($3::text IS NULL OR binding.namespace = $3)
  AND ($4::text = '' OR (binding.namespace, binding.name) > ($4, $5))
ORDER BY binding.namespace, binding.name
LIMIT $6`, scope.kind, scope.id, query.Namespace, query.AfterNamespace, query.AfterName, query.Limit)
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
	if err := validateScope(scope); err != nil {
		return nil, err
	}
	if err := validateRef(ref); err != nil || ref.Revision != nil {
		return nil, ErrInvalidName
	}
	if err := validateVersionPageQuery(query); err != nil {
		return nil, err
	}
	rows, err := r.db.Query(ctx, `
SELECT revision.namespace, revision.name, revision.revision,
       version.media_type, blob.size_bytes,
       revision.revision = binding.current_revision, binding.frozen, revision.created_at
FROM artifact_binding_revisions AS revision
JOIN artifact_bindings AS binding
  ON binding.scope_kind = revision.scope_kind
 AND binding.scope_id = revision.scope_id
 AND binding.namespace = revision.namespace
 AND binding.name = revision.name
JOIN artifact_versions AS version ON version.version_id = revision.version_id
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE revision.scope_kind = $1 AND revision.scope_id = $2
  AND revision.namespace = $3 AND revision.name = $4
  AND ($5::timestamptz IS NULL OR (revision.created_at, revision.revision) < ($5, $6))
ORDER BY revision.created_at DESC, revision.revision DESC
LIMIT $7`, scope.kind, scope.id, ref.Namespace, ref.Name,
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

func (r *PostgresRepository) ListLineage(
	ctx context.Context, scope Scope, exact ArtifactRef, query LineagePageQuery,
) ([]LineageEdge, error) {
	if err := validateScope(scope); err != nil {
		return nil, err
	}
	if _, err := exactRevision(exact); err != nil {
		return nil, err
	}
	if err := validateLineagePageQuery(query); err != nil {
		return nil, err
	}
	rows, err := r.db.Query(ctx, `
SELECT lineage.lineage_kind,
       lineage.source_scope_kind, lineage.source_namespace, lineage.source_name, lineage.source_revision,
       lineage.target_scope_kind, lineage.target_namespace, lineage.target_name, lineage.target_revision,
       lineage.created_at
FROM artifact_lineage AS lineage
WHERE ((
      lineage.source_scope_kind = $1 AND lineage.source_scope_id = $2
      AND lineage.source_namespace = $3 AND lineage.source_name = $4 AND lineage.source_revision = $5
    ) OR (
      lineage.target_scope_kind = $1 AND lineage.target_scope_id = $2
      AND lineage.target_namespace = $3 AND lineage.target_name = $4 AND lineage.target_revision = $5
    ))
  AND ($6::timestamptz IS NULL OR (
    lineage.created_at, lineage.target_revision, lineage.source_revision, lineage.lineage_kind
  ) < ($6, $7, $8, $9))
ORDER BY lineage.created_at DESC, lineage.target_revision DESC,
         lineage.source_revision DESC, lineage.lineage_kind DESC
LIMIT $10`, scope.kind, scope.id, exact.Namespace, exact.Name, *exact.Revision,
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
			&edge.SourceScope, &edge.Source.Namespace, &edge.Source.Name, &sourceRevision,
			&edge.TargetScope, &edge.Target.Namespace, &edge.Target.Name, &targetRevision,
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
		&result.MediaType, &result.Size, &result.Current, &result.Frozen, &result.CreatedAt,
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
