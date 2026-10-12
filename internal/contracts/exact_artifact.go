package contracts

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contentdigest"
)

// ErrExactArtifactMismatch reports that what a store holds at an exact
// artifact revision is not the content its descriptor names. Retrying the
// same read cannot repair it.
var ErrExactArtifactMismatch = errors.New("exact artifact content differs from its descriptor")

// ExactArtifact names one exact artifact revision and the content it holds:
// the sha256 digest of its bytes and, when the holder records them, their
// media type and size. It is the one exact-content descriptor of Audit
// records, finding receipts, retained Audit standards and Audit execution
// inputs; each check of exact content is Verify.
//
// Records that persist only an exact ref and its digest (Audit rounds,
// executions and results) read back as a digest-only descriptor without media
// type and size. The digest still pins the bytes, and therefore the size.
type ExactArtifact struct {
	Ref       ArtifactRef `json:"ref"`
	Digest    string      `json:"digest"`
	MediaType string      `json:"mediaType,omitempty"`
	SizeBytes int64       `json:"sizeBytes,omitempty"`
}

// ExactContent describes bytes read from the exact revision ref.
func ExactContent(ref ArtifactRef, mediaType string, data []byte) ExactArtifact {
	return ExactArtifact{
		Ref: ref.Clone(), Digest: contentdigest.Bytes(data),
		MediaType: mediaType, SizeBytes: int64(len(data)),
	}
}

// HasMetadata reports whether the descriptor records media type and size.
func (e ExactArtifact) HasMetadata() bool { return e.MediaType != "" }

// Validate checks the descriptor's shape: an exact ref, a sha256 digest, and
// either a valid media type with a non-negative size or neither of them.
func (e ExactArtifact) Validate() error {
	if err := e.Ref.ValidateExact(); err != nil {
		return err
	}
	if !contentdigest.Valid(e.Digest) {
		return Invalidf("exact artifact digest must be sha256:<64 lowercase hex>")
	}
	if !e.HasMetadata() {
		if e.SizeBytes != 0 {
			return Invalidf("exact artifact size requires its media type")
		}
		return nil
	}
	return validateArtifactMetadata(e.MediaType, e.SizeBytes)
}

// Verify reports whether found, the descriptor of what a store holds at
// e.Ref, is the content e names: the same exact revision and digest and,
// when e records them, the same media type and size. A digest-only e cannot
// pin a media type. A mismatch is ErrExactArtifactMismatch; an invalid
// descriptor is a contract validation error.
func (e ExactArtifact) Verify(found ExactArtifact) error {
	if err := e.Validate(); err != nil {
		return err
	}
	if !found.Ref.SameExact(e.Ref) || found.Digest != e.Digest {
		return fmt.Errorf("%w: %s/%s", ErrExactArtifactMismatch, e.Ref.Namespace, e.Ref.Name)
	}
	if e.HasMetadata() && (found.MediaType != e.MediaType || found.SizeBytes != e.SizeBytes) {
		return fmt.Errorf("%w: %s/%s metadata", ErrExactArtifactMismatch, e.Ref.Namespace, e.Ref.Name)
	}
	return nil
}

// MarshalJSON writes media type and size together: a descriptor that records
// them carries both, even for empty content, and a digest-only descriptor
// carries neither. A size without a media type is not a descriptor.
func (e ExactArtifact) MarshalJSON() ([]byte, error) {
	if !e.HasMetadata() {
		if e.SizeBytes != 0 {
			return nil, Invalidf("exact artifact size requires its media type")
		}
		return json.Marshal(struct {
			Ref    ArtifactRef `json:"ref"`
			Digest string      `json:"digest"`
		}{e.Ref, e.Digest})
	}
	return json.Marshal(struct {
		Ref       ArtifactRef `json:"ref"`
		Digest    string      `json:"digest"`
		MediaType string      `json:"mediaType"`
		SizeBytes int64       `json:"sizeBytes"`
	}{e.Ref, e.Digest, e.MediaType, e.SizeBytes})
}
