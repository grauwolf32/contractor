package artifacts

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

// MaxExactReadBatchSize bounds the number of references in one metadata query
// (LIM-ART-8). The combined payload of ReadExactBatch is independently
// bounded by MaxPayloadSize; DescribeExact reads no payload.
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
		if errors.Is(err, ErrBlobMissing) {
			removed, recheckErr := r.revisionRemoved(ctx, requests[index].Scope, requests[index].Ref, *results[index].Ref.Revision)
			if recheckErr != nil {
				return nil, recheckErr
			}
			if removed {
				return nil, ErrArtifactNotFound
			}
		}
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
			&result.BindingCreatedAt, &result.RevisionCreatedAt, &totalBytes, &result.GitSource); err != nil {
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

// DescribeExact returns the exact descriptor of each of refs, at most
// MaxExactReadBatchSize exact revisions of scope (ErrPayloadTooLarge
// otherwise, as for ReadExactBatch), in request order, from one statement
// that reads no payload. A missing reference is ErrArtifactNotFound
// for the whole batch, and a stored descriptor that fails exact-artifact
// validation (media type or size) is ErrArtifactIntegrity.
func (r *PostgresRepository) DescribeExact(
	ctx context.Context, scope Scope, refs []ArtifactRef,
) ([]contracts.ExactArtifact, error) {
	if len(refs) > MaxExactReadBatchSize {
		return nil, ErrPayloadTooLarge
	}
	if len(refs) == 0 {
		return []contracts.ExactArtifact{}, nil
	}
	if err := validateScope(scope); err != nil {
		return nil, err
	}
	namespaces, names, revisions := make([]string, len(refs)), make([]string, len(refs)), make([]string, len(refs))
	for index, ref := range refs {
		if err := validateRef(ref); err != nil {
			return nil, err
		}
		if ref.Revision == nil {
			return nil, ErrExactRevisionRequired
		}
		namespaces[index], names[index], revisions[index] = ref.Namespace, ref.Name, *ref.Revision
	}
	rows, err := r.db.Query(ctx, describeExactSQL, scope.kind, scope.id, namespaces, names, revisions)
	if err != nil {
		return nil, fmt.Errorf("describe exact artifact batch: %w", err)
	}
	defer rows.Close()
	result := make([]contracts.ExactArtifact, 0, len(refs))
	for rows.Next() {
		var ordinal int64
		var described contracts.ExactArtifact
		if err := rows.Scan(&ordinal, &described.MediaType, &described.SizeBytes, &described.Digest); err != nil {
			return nil, fmt.Errorf("scan exact artifact description: %w", err)
		}
		if ordinal < 1 || ordinal > int64(len(refs)) {
			return nil, ErrArtifactIntegrity
		}
		if ordinal != int64(len(result)+1) {
			return nil, ErrArtifactNotFound
		}
		described.Ref = refs[len(result)].Clone()
		if described.Validate() != nil {
			return nil, ErrArtifactIntegrity
		}
		result = append(result, described)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate exact artifact descriptions: %w", err)
	}
	if len(result) != len(refs) {
		return nil, ErrArtifactNotFound
	}
	return result, nil
}
