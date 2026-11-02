package postgres

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
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
func reusableBlobKey(ctx context.Context, db postgres.DBTX, candidate blob.Object) (*string, error) {
	if candidate.Backend != blob.Filesystem && candidate.Backend != blob.Postgres {
		return nil, blob.ErrIntegrity
	}
	store := activeBlobStore(ctx)
	// A preprepared payload may have lost its file before reaching publication.
	// Never replace a missing reference with another unreadable candidate.
	// Store derived the candidate's digest from the bytes it wrote, so only its
	// presence and size need rechecking; this keeps every write from re-reading
	// up to 64 MiB inside the caller's transaction.
	if candidate.Backend == blob.Filesystem {
		if err := store.Check(ctx, candidate); err != nil {
			return nil, err
		}
	}
	var backend blob.Backend
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
	if size != candidate.Size {
		return nil, blob.ErrIntegrity
	}
	if backend == blob.Postgres {
		if key != nil {
			return nil, blob.ErrIntegrity
		}
		// Inline bytes are checksum-validated by PostgreSQL. Keep this existing
		// location even when the new-write policy prepared a file candidate.
		return nil, nil
	}
	if backend != blob.Filesystem || key == nil {
		return nil, blob.ErrIntegrity
	}
	if *key == candidate.Key {
		return key, nil
	}
	existing := blob.Object{Backend: backend, Key: *key, Digest: candidate.Digest, Size: size}
	if _, err := store.Read(ctx, existing); errors.Is(err, blob.ErrMissing) {
		if candidate.Backend == blob.Filesystem {
			return nil, nil
		}
		return nil, err
	} else if err != nil {
		// Corruption, permissions and cancellation are not evidence of absence.
		return nil, err
	}
	return key, nil
}

func nullableBlobKey(key string) any {
	if key == "" {
		return nil
	}
	return key
}
