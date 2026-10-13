package artifacts

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"path/filepath"

	"github.com/grauwolf32/contractor/internal/apierror"
)

type BlobBackend string

const (
	// BlobAuto selects the new-write policy; each blob records its own backend.
	BlobAuto       BlobBackend = "auto"
	BlobPostgres   BlobBackend = "postgresql"
	BlobFilesystem BlobBackend = "filesystem"
)

const DefaultBlobInlineMaxBytes int64 = 1 << 20

var (
	ErrBlobBackend = errors.New("invalid or unsupported artifact blob backend")
	ErrBlobMissing = apierror.New(apierror.ArtifactContentMissing, "artifact blob content is missing")
	// ErrBlobIO reports a failed blob storage operation. Unlike
	// ErrArtifactIntegrity it proves nothing about the stored bytes, and the
	// same operation may succeed on retry.
	ErrBlobIO = errors.New("artifact blob storage I/O failed")
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
	// Check confirms that a stored object is still present with its recorded
	// size without reading its payload; only Read verifies the digest.
	Check(context.Context, BlobObject) error
	Delete(context.Context, BlobObject) error
}

func ValidateBlobConfig(kind, path string) (BlobBackend, error) {
	if kind == "" {
		kind = string(BlobAuto)
	}
	switch BlobBackend(kind) {
	case BlobAuto, BlobPostgres:
	case BlobFilesystem:
		if !filepath.IsAbs(path) || filepath.Clean(path) != path {
			return "", fmt.Errorf("%w: filesystem requires a clean absolute blob path", ErrBlobBackend)
		}
	default:
		return "", fmt.Errorf("%w: %s", ErrBlobBackend, kind)
	}
	if path != "" && (!filepath.IsAbs(path) || filepath.Clean(path) != path) {
		return "", fmt.Errorf("%w: blob path must be clean and absolute", ErrBlobBackend)
	}
	return BlobBackend(kind), nil
}

// ValidateBlobInlineLimit checks the inclusive inline threshold for auto mode.
func ValidateBlobInlineLimit(value int64) error {
	if value < 0 || value > MaxPayloadSize {
		return fmt.Errorf("%w: inline byte limit must be between 0 and %d", ErrBlobBackend, MaxPayloadSize)
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

// Inline bytes travel with the object; Read verifies them.
func (PostgresBlobStore) Check(ctx context.Context, object BlobObject) error {
	if object.Backend != BlobPostgres || object.Key != "" {
		return ErrArtifactIntegrity
	}
	return ctx.Err()
}

// Deletion of inline bytes is owned by the registry DELETE, never a second I/O.
func (PostgresBlobStore) Delete(ctx context.Context, object BlobObject) error {
	if object.Backend != BlobPostgres {
		return ErrArtifactIntegrity
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
