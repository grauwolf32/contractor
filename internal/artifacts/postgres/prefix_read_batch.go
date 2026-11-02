package postgres

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/jackc/pgx/v5"
)

func (r *Repository) ReadPrefixBatch(
	ctx context.Context, scope artifacts.Scope, namespace, prefix string, limit int,
) ([]artifacts.ReadResult, error) {
	ctx, releaseTransfer, err := r.AcquireTransfer(ctx)
	if err != nil {
		return nil, err
	}
	defer releaseTransfer()
	results, err := r.readPrefixBatch(ctx, scope, namespace, prefix, limit)
	if errors.Is(err, errRevisionRemoved) {
		// History retention removed a revision that was current when the
		// metadata was read, after a newer write; read the batch once more.
		results, err = r.readPrefixBatch(ctx, scope, namespace, prefix, limit)
	}
	if errors.Is(err, errRevisionRemoved) {
		return nil, fmt.Errorf("%w: current revisions kept moving", artifacts.ErrHistoryMaintenancePending)
	}
	return results, err
}

func (r *Repository) readPrefixBatch(
	ctx context.Context, scope artifacts.Scope, namespace, prefix string, limit int,
) ([]artifacts.ReadResult, error) {
	// Limit bindings before joining versions and blobs. LEFT JOINs make a
	// dangling current revision or missing blob an integrity error rather than
	// silently dropping a binding from the notebook.
	rows, err := r.db.Query(ctx, readPrefixBatchSQL, scope.Kind(), scope.ID(), namespace, prefix, limit, artifacts.MaxPayloadSize)
	if err != nil {
		return nil, fmt.Errorf("read artifact prefix batch: %w", err)
	}
	results, objects, err := scanPrefixReadBatch(rows, limit)
	if err != nil {
		return nil, err
	}
	for index, object := range objects {
		data, err := r.runtime.Store.Read(ctx, object)
		if errors.Is(err, blob.ErrMissing) {
			removed, recheckErr := r.revisionRemoved(ctx, scope, results[index].Ref, *results[index].Ref.Revision)
			if recheckErr != nil {
				return nil, recheckErr
			}
			if removed {
				return nil, errRevisionRemoved
			}
		}
		if err != nil {
			return nil, err
		}
		results[index].Payload.Data = data
	}
	return results, nil
}

func scanPrefixReadBatch(rows pgx.Rows, limit int) ([]artifacts.ReadResult, []blob.Object, error) {
	defer rows.Close()
	results := make([]artifacts.ReadResult, 0, limit)
	objects := make([]blob.Object, 0, limit)
	for rows.Next() {
		var result artifacts.ReadResult
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
			return nil, nil, blob.ErrIntegrity
		}
		if *totalBytes > artifacts.MaxPayloadSize {
			return nil, nil, blob.ErrTooLarge
		}
		if artifacts.ValidateMediaType(*mediaType) != nil {
			return nil, nil, blob.ErrIntegrity
		}
		result.Ref.Revision = revision
		result.Payload.MediaType = *mediaType
		result.RevisionCreatedAt = *revisionCreatedAt
		object := blob.Object{Backend: blob.Backend(*backend), Inline: inline, Digest: digest, Size: *size}
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
