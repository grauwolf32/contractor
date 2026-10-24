package artifacts

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

func (r *PostgresRepository) Read(ctx context.Context, scope Scope, ref ArtifactRef) (ReadResult, error) {
	ctx, releaseTransfer, err := AcquireTransfer(ctx)
	if err != nil {
		return ReadResult{}, err
	}
	defer releaseTransfer()
	result, err := r.read(ctx, scope, ref)
	if !errors.Is(err, errRevisionRemoved) {
		return result, err
	}
	if ref.Revision != nil {
		return ReadResult{}, ErrArtifactNotFound
	}
	// History retention removed the revision that was current when the
	// metadata was read, after a newer write; read the new current once.
	result, err = r.read(ctx, scope, ref)
	if errors.Is(err, errRevisionRemoved) {
		return ReadResult{}, fmt.Errorf("%w: the current revision kept moving", ErrHistoryMaintenancePending)
	}
	return result, err
}

// errRevisionRemoved reports content that was missing because its revision
// was removed after its metadata was read.
var errRevisionRemoved = errors.New("artifact revision was removed while it was read")

func (r *PostgresRepository) read(ctx context.Context, scope Scope, ref ArtifactRef) (ReadResult, error) {
	query := readBindingSQL
	arguments := []any{scope.kind, scope.id, ref.Namespace, ref.Name}
	if ref.Revision != nil {
		query = readRevisionSQL
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
	err := r.db.QueryRow(ctx, query, arguments...).Scan(
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
	if errors.Is(err, ErrBlobMissing) {
		if removed, recheckErr := r.revisionRemoved(ctx, scope, ref, revision); recheckErr != nil {
			return ReadResult{}, recheckErr
		} else if removed {
			return ReadResult{}, errRevisionRemoved
		}
	}
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
