package artifacts

import (
	"context"
	"errors"
	"fmt"

	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// reusableBlobKey verifies physical bytes, not merely the SHA registry entry.
// The publication statement may reuse ONLY this exact key. If another writer
// changes the row after this check, publication uses its own complete candidate
// instead. This preserves the repository's caller-owned transaction boundary
// and avoids a check-then-unconditionally-reuse race with publication/purge.
// A displaced generation is left for offline cleanup, not unlinked here:
// readers may already have resolved it before the competing row update.
func reusableBlobKey(ctx context.Context, db postgres.DBTX, candidate BlobObject) (*string, error) {
	if candidate.Backend != BlobFilesystem {
		return nil, nil
	}
	store := activeBlobStore(ctx)
	// A preprepared payload may have lost its file before reaching publication.
	// Never replace a missing reference with another unreadable candidate.
	if _, err := store.Read(ctx, candidate); err != nil {
		return nil, err
	}
	var backend BlobBackend
	var key *string
	var size int64
	err := db.QueryRow(ctx, `SELECT backend, object_key, size_bytes
FROM artifact_blobs WHERE sha256 = $1`, candidate.Digest).Scan(&backend, &key, &size)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("inspect deduplicated artifact blob: %w", err)
	}
	if backend != BlobFilesystem || key == nil || size != candidate.Size {
		return nil, ErrArtifactIntegrity
	}
	if *key == candidate.Key {
		return key, nil
	}
	existing := BlobObject{Backend: backend, Key: *key, Digest: candidate.Digest, Size: size}
	if _, err := store.Read(ctx, existing); errors.Is(err, ErrBlobMissing) {
		return nil, nil
	} else if err != nil {
		// Corruption, permissions and cancellation are not evidence of absence.
		return nil, err
	}
	return key, nil
}
