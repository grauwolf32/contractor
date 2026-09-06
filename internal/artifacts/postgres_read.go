package artifacts

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

func (r *PostgresRepository) Read(ctx context.Context, scope Scope, ref ArtifactRef) (ReadResult, error) {
	if err := validateScope(scope); err != nil {
		return ReadResult{}, err
	}
	if err := validateRef(ref); err != nil {
		return ReadResult{}, err
	}
	ctx, releaseTransfer, err := AcquireTransfer(ctx)
	if err != nil {
		return ReadResult{}, err
	}
	defer releaseTransfer()
	query := `
SELECT revision.revision, version.media_type, blob.backend, blob.object_key, blob.payload, blob.sha256, blob.size_bytes,
       binding.created_at, revision.created_at
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
  AND binding.namespace = $3 AND binding.name = $4`
	arguments := []any{scope.kind, scope.id, ref.Namespace, ref.Name}
	if ref.Revision != nil {
		query = `
SELECT revision.revision, version.media_type, blob.backend, blob.object_key, blob.payload, blob.sha256, blob.size_bytes,
       binding.created_at, revision.created_at
FROM artifact_binding_revisions AS revision
JOIN artifact_bindings AS binding
  ON binding.scope_kind = revision.scope_kind
 AND binding.scope_id = revision.scope_id
 AND binding.namespace = revision.namespace
 AND binding.name = revision.name
JOIN artifact_versions AS version ON version.version_id = revision.version_id
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE revision.scope_kind = $1 AND revision.scope_id = $2
  AND revision.namespace = $3 AND revision.name = $4 AND revision.revision = $5`
		arguments = append(arguments, *ref.Revision)
	}

	var revision string
	var mediaType string
	var backend BlobBackend
	var objectKey *string
	var data []byte
	var storedDigest []byte
	var size int64
	var bindingCreatedAt time.Time
	var revisionCreatedAt time.Time
	err = r.db.QueryRow(ctx, query, arguments...).Scan(
		&revision, &mediaType, &backend, &objectKey, &data, &storedDigest, &size,
		&bindingCreatedAt, &revisionCreatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return ReadResult{}, ErrArtifactNotFound
	}
	if err != nil {
		return ReadResult{}, fmt.Errorf("read artifact %s/%s: %w", ref.Namespace, ref.Name, err)
	}
	object := BlobObject{Backend: backend, Inline: data, Digest: storedDigest, Size: size}
	if objectKey != nil {
		object.Key = *objectKey
	}
	data, err = activeBlobStore(ctx).Read(ctx, object)
	if err != nil {
		return ReadResult{}, err
	}
	if validateMediaType(mediaType) != nil {
		return ReadResult{}, ErrArtifactIntegrity
	}
	exact := revision
	return ReadResult{
		Ref:               ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &exact},
		Payload:           Payload{MediaType: mediaType, Data: data},
		BindingCreatedAt:  bindingCreatedAt,
		RevisionCreatedAt: revisionCreatedAt,
	}, nil
}

func (r *PostgresRepository) List(
	ctx context.Context,
	scope Scope,
	namespace *string,
) ([]ArtifactRef, error) {
	if err := validateScope(scope); err != nil {
		return nil, err
	}
	if namespace != nil {
		if err := validateComponent(*namespace); err != nil {
			return nil, err
		}
	}
	rows, err := r.db.Query(ctx, `
SELECT namespace, name
FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2
  AND ($3::text IS NULL OR namespace = $3)
ORDER BY namespace, name`, scope.kind, scope.id, namespace)
	if err != nil {
		return nil, fmt.Errorf("list artifacts: %w", err)
	}
	defer rows.Close()
	result := make([]ArtifactRef, 0)
	for rows.Next() {
		var ref ArtifactRef
		if err := rows.Scan(&ref.Namespace, &ref.Name); err != nil {
			return nil, fmt.Errorf("scan artifact binding: %w", err)
		}
		result = append(result, ref)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate artifact bindings: %w", err)
	}
	return result, nil
}
