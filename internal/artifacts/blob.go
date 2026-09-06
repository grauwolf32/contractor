package artifacts

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"path/filepath"

	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type BlobBackend string

const (
	BlobPostgres   BlobBackend = "postgresql"
	BlobFilesystem BlobBackend = "filesystem"
)

var (
	ErrBlobBackend         = errors.New("invalid or unsupported artifact blob backend")
	ErrBlobBackendMismatch = errors.New("artifact blob backend does not match the installation")
	ErrBlobMissing         = errors.New("artifact blob content is missing")
)

// BlobObject describes physical bytes; it never crosses the Artifact API.
// Inline is committed/deleted by the PostgreSQL registry transaction itself.
type BlobObject struct {
	Backend BlobBackend
	Key     string
	Digest  []byte
	Size    int64
	Inline  []byte
}

// BlobStore prepares complete immutable bytes before their registry publication.
// The inline adapter returns bytes for the caller's existing SQL transaction.
type BlobStore interface {
	Store(context.Context, []byte) (BlobObject, error)
	Read(context.Context, BlobObject) ([]byte, error)
	Delete(context.Context, BlobObject) error
}

func ValidateBlobConfig(kind, path string) (BlobBackend, error) {
	if kind == "" {
		kind = string(BlobPostgres)
	}
	switch BlobBackend(kind) {
	case BlobPostgres:
		if path != "" {
			return "", fmt.Errorf("%w: postgresql does not accept a blob path", ErrBlobBackend)
		}
	case BlobFilesystem:
		if !filepath.IsAbs(path) || filepath.Clean(path) != path {
			return "", fmt.Errorf("%w: filesystem requires a clean absolute blob path", ErrBlobBackend)
		}
	default:
		return "", fmt.Errorf("%w: %s", ErrBlobBackend, kind)
	}
	return BlobBackend(kind), nil
}

// ClaimBlobBackend serializes first startup and prevents implicit migration.
func ClaimBlobBackend(ctx context.Context, db postgres.DBTX, kind BlobBackend) error {
	if kind != BlobPostgres && kind != BlobFilesystem {
		return ErrBlobBackend
	}
	var selected string
	err := db.QueryRow(ctx, `UPDATE artifact_blob_settings SET backend = $1
WHERE singleton AND (backend IS NULL OR backend = $1) RETURNING backend`, string(kind)).Scan(&selected)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrBlobBackendMismatch
	}
	if err != nil {
		return fmt.Errorf("claim artifact blob backend: %w", err)
	}
	return nil
}

type PostgresBlobStore struct{}

func (PostgresBlobStore) Store(ctx context.Context, data []byte) (BlobObject, error) {
	if err := ctx.Err(); err != nil {
		return BlobObject{}, err
	}
	if len(data) > MaxPayloadSize {
		return BlobObject{}, ErrPayloadTooLarge
	}
	digest := sha256.Sum256(data)
	if data == nil {
		data = []byte{}
	}
	return BlobObject{Backend: BlobPostgres, Digest: digest[:], Size: int64(len(data)), Inline: data}, nil
}

func (PostgresBlobStore) Read(ctx context.Context, object BlobObject) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if object.Backend != BlobPostgres || object.Key != "" {
		return nil, ErrArtifactIntegrity
	}
	if err := verifyBlob(object, object.Inline); err != nil {
		return nil, err
	}
	return object.Inline, nil
}

// Deletion of inline bytes is owned by the registry DELETE, never a second I/O.
func (PostgresBlobStore) Delete(ctx context.Context, object BlobObject) error {
	if object.Backend != BlobPostgres {
		return ErrBlobBackendMismatch
	}
	return ctx.Err()
}

func verifyBlob(object BlobObject, data []byte) error {
	if object.Size < 0 || object.Size > MaxPayloadSize || int64(len(data)) != object.Size {
		return ErrArtifactIntegrity
	}
	digest := sha256.Sum256(data)
	if !bytes.Equal(object.Digest, digest[:]) {
		return ErrArtifactIntegrity
	}
	return nil
}

func nullableBlobKey(key string) any {
	if key == "" {
		return nil
	}
	return key
}
