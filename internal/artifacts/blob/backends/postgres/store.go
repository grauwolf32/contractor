package postgres

import (
	"context"
	"crypto/sha256"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type Store struct{}

func (Store) Store(ctx context.Context, data []byte) (blob.Object, error) {
	if err := ctx.Err(); err != nil {
		return blob.Object{}, err
	}
	if len(data) > contracts.MaxArtifactPayloadBytes {
		return blob.Object{}, blob.ErrTooLarge
	}
	digest := sha256.Sum256(data)
	if data == nil {
		data = []byte{}
	}
	return blob.Object{Backend: blob.Postgres, Digest: digest[:], Size: int64(len(data)), Inline: data}, nil
}

func (Store) Read(ctx context.Context, object blob.Object) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if object.Backend != blob.Postgres || object.Key != "" {
		return nil, blob.ErrIntegrity
	}
	if err := blob.Verify(object, object.Inline); err != nil {
		return nil, err
	}
	return object.Inline, nil
}

// Inline bytes travel with the object; Read verifies them.
func (Store) Check(ctx context.Context, object blob.Object) error {
	if object.Backend != blob.Postgres || object.Key != "" {
		return blob.ErrIntegrity
	}
	return ctx.Err()
}

// Deletion of inline bytes is owned by the registry DELETE, never a second I/O.
func (Store) Delete(ctx context.Context, object blob.Object) error {
	if object.Backend != blob.Postgres {
		return blob.ErrIntegrity
	}
	return ctx.Err()
}
