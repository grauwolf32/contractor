package artifacts

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// MaxExactReadBatchSize bounds the number of references in one metadata query.
// The combined payload is independently bounded by MaxPayloadSize.
const MaxExactReadBatchSize = 32

// ExactReadRequest carries a scope selected by the trusted caller, independently
// of the artifact reference. Authorization remains with that caller.
type ExactReadRequest struct {
	Scope Scope
	Ref   ArtifactRef
}

// ReadExactBatch reads immutable revisions in request order through the same
// blob adapter and transfer admission used by Read. It returns no partial result
// when a reference is missing or any payload fails integrity validation.
func (r *PostgresRepository) ReadExactBatch(ctx context.Context, requests []ExactReadRequest) ([]ReadResult, error) {
	if len(requests) > MaxExactReadBatchSize {
		return nil, ErrPayloadTooLarge
	}
	if len(requests) == 0 {
		return []ReadResult{}, nil
	}
	kinds, ids := make([]string, len(requests)), make([]string, len(requests))
	namespaces, names, revisions := make([]string, len(requests)), make([]string, len(requests)), make([]string, len(requests))
	for index, request := range requests {
		if err := validateScope(request.Scope); err != nil {
			return nil, err
		}
		if err := validateRef(request.Ref); err != nil {
			return nil, err
		}
		if request.Ref.Revision == nil {
			return nil, ErrExactRevisionRequired
		}
		kinds[index], ids[index] = string(request.Scope.kind), request.Scope.id
		namespaces[index], names[index], revisions[index] = request.Ref.Namespace, request.Ref.Name, *request.Ref.Revision
	}
	ctx, releaseTransfer, err := AcquireTransfer(ctx)
	if err != nil {
		return nil, err
	}
	defer releaseTransfer()
	rows, err := r.db.Query(ctx, readExactBatchSQL, kinds, ids, namespaces, names, revisions, MaxPayloadSize)
	if err != nil {
		return nil, fmt.Errorf("read exact artifact batch: %w", err)
	}
	results, objects, err := scanExactReadBatch(rows, requests)
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

// Release the database connection before resolving filesystem payloads.
func scanExactReadBatch(rows pgx.Rows, requests []ExactReadRequest) ([]ReadResult, []BlobObject, error) {
	defer rows.Close()
	results := make([]ReadResult, 0, len(requests))
	objects := make([]BlobObject, 0, len(requests))
	for rows.Next() {
		var result ReadResult
		var object BlobObject
		var objectKey *string
		var ordinal, totalBytes int64
		var revision string
		if err := rows.Scan(&ordinal, &revision, &result.Payload.MediaType,
			&object.Backend, &objectKey, &object.Inline, &object.Digest, &object.Size,
			&result.BindingCreatedAt, &result.RevisionCreatedAt, &totalBytes); err != nil {
			return nil, nil, fmt.Errorf("scan exact artifact batch: %w", err)
		}
		if totalBytes > MaxPayloadSize {
			return nil, nil, ErrPayloadTooLarge
		}
		if ordinal != int64(len(results)+1) {
			return nil, nil, ErrArtifactNotFound
		}
		if ordinal > int64(len(requests)) || validateMediaType(result.Payload.MediaType) != nil {
			return nil, nil, ErrArtifactIntegrity
		}
		if objectKey != nil {
			object.Key = *objectKey
		}
		ref := requests[len(results)].Ref
		result.Ref = ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &revision}
		results = append(results, result)
		objects = append(objects, object)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("iterate exact artifact batch: %w", err)
	}
	if len(results) != len(requests) {
		return nil, nil, ErrArtifactNotFound
	}
	return results, objects, nil
}
