package artifacts

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

type prefixReadBatchRepository interface {
	ReadPrefixBatch(context.Context, Scope, string, string, int) ([]ReadResult, error)
}

// ReadPrefixBatch returns at most limit current bindings and their payloads in
// name order. It lets callers detect quota overflow by requesting one extra
// binding without performing a separate listing and one read per artifact.
func (s ScopedStore) ReadPrefixBatch(ctx context.Context, namespace, prefix string, limit int) ([]ReadResult, error) {
	if err := validateScope(s.scope); err != nil {
		return nil, err
	}
	if err := validatePrefixList(namespace, prefix, limit); err != nil {
		return nil, err
	}
	repository, ok := s.service.repository.(prefixReadBatchRepository)
	if !ok {
		return nil, ErrQueryUnsupported
	}
	return repository.ReadPrefixBatch(ctx, s.scope, namespace, prefix, limit)
}

func (r *PostgresRepository) ReadPrefixBatch(
	ctx context.Context, scope Scope, namespace, prefix string, limit int,
) ([]ReadResult, error) {
	if err := validateScope(scope); err != nil {
		return nil, err
	}
	if err := validatePrefixList(namespace, prefix, limit); err != nil {
		return nil, err
	}
	ctx, releaseTransfer, err := AcquireTransfer(ctx)
	if err != nil {
		return nil, err
	}
	defer releaseTransfer()

	// Limit bindings before joining versions and blobs. LEFT JOINs make a
	// dangling current revision or missing blob an integrity error rather than
	// silently dropping a binding from the notebook.
	rows, err := r.db.Query(ctx, `
WITH limited AS MATERIALIZED (
    SELECT scope_kind, scope_id, namespace, name, current_revision, created_at
    FROM artifact_bindings
    WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3
      AND starts_with(name, $4)
    ORDER BY name
    LIMIT $5
)
SELECT binding.namespace, binding.name, revision.revision, version.media_type,
       blob.backend, blob.object_key,
       CASE WHEN sum(blob.size_bytes) OVER () <= $6 THEN blob.payload END,
       blob.sha256, blob.size_bytes, binding.created_at, revision.created_at,
       sum(blob.size_bytes) OVER ()
FROM limited AS binding
LEFT JOIN artifact_binding_revisions AS revision
  ON revision.scope_kind = binding.scope_kind AND revision.scope_id = binding.scope_id
 AND revision.namespace = binding.namespace AND revision.name = binding.name
 AND revision.revision = binding.current_revision
LEFT JOIN artifact_versions AS version ON version.version_id = revision.version_id
LEFT JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
ORDER BY binding.name`, scope.kind, scope.id, namespace, prefix, limit, MaxPayloadSize)
	if err != nil {
		return nil, fmt.Errorf("read artifact prefix batch: %w", err)
	}
	results, objects, err := scanPrefixReadBatch(rows, limit)
	if err != nil {
		return nil, err
	}
	for index, object := range objects {
		data, err := activeBlobStore(ctx).Read(ctx, object)
		if err != nil {
			return nil, err
		}
		results[index].Payload.Data = data
	}
	return results, nil
}

func scanPrefixReadBatch(rows pgx.Rows, limit int) ([]ReadResult, []BlobObject, error) {
	defer rows.Close()
	results := make([]ReadResult, 0, limit)
	objects := make([]BlobObject, 0, limit)
	for rows.Next() {
		var result ReadResult
		var revision, mediaType, backend, objectKey *string
		var inline, digest []byte
		var size, totalBytes *int64
		var revisionCreatedAt *time.Time
		if err := rows.Scan(&result.Ref.Namespace, &result.Ref.Name, &revision, &mediaType,
			&backend, &objectKey, &inline, &digest, &size, &result.BindingCreatedAt,
			&revisionCreatedAt, &totalBytes); err != nil {
			return nil, nil, fmt.Errorf("scan artifact prefix batch: %w", err)
		}
		if revision == nil || mediaType == nil || backend == nil || size == nil ||
			revisionCreatedAt == nil || totalBytes == nil {
			return nil, nil, ErrArtifactIntegrity
		}
		if *totalBytes > MaxPayloadSize {
			return nil, nil, ErrPayloadTooLarge
		}
		if validateMediaType(*mediaType) != nil {
			return nil, nil, ErrArtifactIntegrity
		}
		result.Ref.Revision = revision
		result.Payload.MediaType = *mediaType
		result.RevisionCreatedAt = *revisionCreatedAt
		object := BlobObject{Backend: BlobBackend(*backend), Inline: inline, Digest: digest, Size: *size}
		if objectKey != nil {
			object.Key = *objectKey
		}
		results = append(results, result)
		objects = append(objects, object)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("iterate artifact prefix batch: %w", err)
	}
	return results, objects, nil
}
