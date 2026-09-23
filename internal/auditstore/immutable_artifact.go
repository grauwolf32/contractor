package auditstore

import (
	"bytes"
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// WriteImmutableArtifact writes a content-named Audit artifact once. A crash
// after the write but before its binding is recorded is recovered by accepting
// an existing artifact only when its media type and bytes are identical; any
// other existing content returns collision, so each caller keeps its own error
// mapping.
func WriteImmutableArtifact(
	ctx context.Context,
	store artifacts.ScopedStore,
	target contracts.ArtifactRef,
	payload artifacts.Payload,
	collision error,
) (ExactArtifact, error) {
	written, err := store.Write(ctx, target, payload, nil)
	if err == nil {
		return ExactArtifact{
			Ref: written.Ref, Digest: auditdomain.DigestBytes(payload.Data),
			MediaType: written.MediaType, SizeBytes: written.Size,
		}, nil
	}
	if !errors.Is(err, artifacts.ErrArtifactConflict) {
		return ExactArtifact{}, err
	}
	current, readErr := store.Read(ctx, target)
	if readErr != nil {
		return ExactArtifact{}, readErr
	}
	if current.Payload.MediaType != payload.MediaType || !bytes.Equal(current.Payload.Data, payload.Data) {
		return ExactArtifact{}, collision
	}
	return ExactArtifact{
		Ref: current.Ref, Digest: auditdomain.DigestBytes(current.Payload.Data),
		MediaType: current.Payload.MediaType, SizeBytes: int64(len(current.Payload.Data)),
	}, nil
}
