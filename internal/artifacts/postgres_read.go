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
