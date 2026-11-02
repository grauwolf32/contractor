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

func (r *Repository) Read(ctx context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef) (artifacts.ReadResult, error) {
	ctx, releaseTransfer, err := blob.AcquireTransfer(ctx)
	if err != nil {
		return artifacts.ReadResult{}, err
	}
	defer releaseTransfer()
	result, err := r.read(ctx, scope, ref)
	if !errors.Is(err, errRevisionRemoved) {
		return result, err
	}
	if ref.Revision != nil {
		return artifacts.ReadResult{}, artifacts.ErrArtifactNotFound
	}
	// History retention removed the revision that was current when the
	// metadata was read, after a newer write; read the new current once.
	result, err = r.read(ctx, scope, ref)
	if errors.Is(err, errRevisionRemoved) {
		return artifacts.ReadResult{}, fmt.Errorf("%w: the current revision kept moving", artifacts.ErrHistoryMaintenancePending)
	}
	return result, err
}

// errRevisionRemoved reports content that was missing because its revision
// was removed after its metadata was read.
var errRevisionRemoved = errors.New("artifact revision was removed while it was read")

func (r *Repository) read(ctx context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef) (artifacts.ReadResult, error) {
	query := readBindingSQL
	arguments := []any{scope.Kind(), scope.ID(), ref.Namespace, ref.Name}
	if ref.Revision != nil {
		query = readRevisionSQL
		arguments = append(arguments, *ref.Revision)
	}

	var revision string
	var mediaType string
	var backend blob.Backend
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
		return artifacts.ReadResult{}, artifacts.ErrArtifactNotFound
	}
	if err != nil {
		return artifacts.ReadResult{}, fmt.Errorf("read artifact %s/%s: %w", ref.Namespace, ref.Name, err)
	}
	object := blob.Object{Backend: backend, Inline: data, Digest: storedDigest, Size: size}
	if objectKey != nil {
		object.Key = *objectKey
	}
	data, err = activeBlobStore(ctx).Read(ctx, object)
	if errors.Is(err, blob.ErrMissing) {
		if removed, recheckErr := r.revisionRemoved(ctx, scope, ref, revision); recheckErr != nil {
			return artifacts.ReadResult{}, recheckErr
		} else if removed {
			return artifacts.ReadResult{}, errRevisionRemoved
		}
	}
	if err != nil {
		return artifacts.ReadResult{}, err
	}
	if artifacts.ValidateMediaType(mediaType) != nil {
		return artifacts.ReadResult{}, blob.ErrIntegrity
	}
	exact := revision
	return artifacts.ReadResult{
		Ref:               artifacts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &exact},
		Payload:           artifacts.Payload{MediaType: mediaType, Data: data},
		BindingCreatedAt:  bindingCreatedAt,
		RevisionCreatedAt: revisionCreatedAt,
	}, nil
}
