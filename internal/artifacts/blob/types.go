package blob

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"path/filepath"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type Backend string

const (
	// Auto selects the new-write policy; each blob records its own backend.
	Auto       Backend = "auto"
	Postgres   Backend = "postgresql"
	Filesystem Backend = "filesystem"
)

const DefaultInlineMaxBytes int64 = 1 << 20

var (
	ErrIntegrity = apierror.New(apierror.ArtifactContentCorrupt, "artifact payload integrity failure")
	ErrTooLarge  = apierror.New(apierror.ArtifactTooLarge, "artifact payload exceeds size limit")
	ErrBackend   = errors.New("invalid or unsupported artifact blob backend")
	ErrMissing   = apierror.New(apierror.ArtifactContentMissing, "artifact blob content is missing")
	// ErrIO reports a failed blob storage operation. Unlike
	// ErrIntegrity it proves nothing about the stored bytes, and the
	// same operation may succeed on retry.
	ErrIO = errors.New("artifact blob storage I/O failed")
)

// Object describes physical bytes; it never crosses the Artifact API.
// Inline is committed/deleted by the PostgreSQL registry transaction itself.
type Object struct {
	Backend Backend
	Key     string
	Digest  []byte
	Size    int64
	Inline  []byte
}

// Storage prepares complete immutable bytes before their registry publication.
// The inline adapter returns bytes for the caller's existing SQL transaction.
type Storage interface {
	Store(context.Context, []byte) (Object, error)
	Read(context.Context, Object) ([]byte, error)
	// Check confirms that a stored object is still present with its recorded
	// size without reading its payload; only Read verifies the digest.
	Check(context.Context, Object) error
	Delete(context.Context, Object) error
}

func ValidateConfig(kind, path string) (Backend, error) {
	if kind == "" {
		kind = string(Auto)
	}
	switch Backend(kind) {
	case Auto, Postgres:
	case Filesystem:
		if !filepath.IsAbs(path) || filepath.Clean(path) != path {
			return "", fmt.Errorf("%w: filesystem requires a clean absolute blob path", ErrBackend)
		}
	default:
		return "", fmt.Errorf("%w: %s", ErrBackend, kind)
	}
	if path != "" && (!filepath.IsAbs(path) || filepath.Clean(path) != path) {
		return "", fmt.Errorf("%w: blob path must be clean and absolute", ErrBackend)
	}
	return Backend(kind), nil
}

// ValidateInlineLimit checks the inclusive inline threshold for auto mode.
func ValidateInlineLimit(value int64) error {
	if value < 0 || value > contracts.MaxArtifactPayloadBytes {
		return fmt.Errorf("%w: inline byte limit must be between 0 and %d", ErrBackend, contracts.MaxArtifactPayloadBytes)
	}
	return nil
}

// Verify checks size and digest against the complete immutable payload.
func Verify(object Object, data []byte) error {
	if object.Size < 0 || object.Size > contracts.MaxArtifactPayloadBytes || int64(len(data)) != object.Size {
		return ErrIntegrity
	}
	digest := sha256.Sum256(data)
	if !bytes.Equal(object.Digest, digest[:]) {
		return ErrIntegrity
	}
	return nil
}

// Prepared carries a verified candidate and the only existing generation a
// publication may reuse. It is internal state, never client-supplied metadata.
type Prepared struct {
	object      Object
	reusableKey *string
}

func NewPrepared(object Object, key *string) *Prepared {
	return &Prepared{object: object, reusableKey: key}
}
func (p *Prepared) Object() Object { return p.object }
func (p *Prepared) ReusableKey() *string {
	if p.reusableKey == nil {
		return nil
	}
	key := *p.reusableKey
	return &key
}
