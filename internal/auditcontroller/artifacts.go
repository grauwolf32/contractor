package auditcontroller

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// ArtifactAccess keeps package construction testable while production uses
// the existing ProjectScope artifact authority.
type ArtifactAccess interface {
	PutImmutableProject(
		context.Context, string, contracts.ArtifactRef, artifacts.Payload,
	) (auditstore.ExactArtifact, error)
	ResolveProjectExact(
		context.Context, string, auditstore.ExactArtifact,
	) (auditstore.ExactArtifact, error)
	ReadProjectExact(
		context.Context, string, auditstore.ExactArtifact,
	) (artifacts.Payload, error)
}

func (a *ProjectArtifactAccess) ReadProjectExact(
	ctx context.Context,
	projectID string,
	descriptor auditstore.ExactArtifact,
) (artifacts.Payload, error) {
	if err := descriptor.Ref.ValidateExact(); err != nil || descriptor.Digest == "" ||
		descriptor.MediaType == "" || descriptor.SizeBytes < 0 {
		return artifacts.Payload{}, fmt.Errorf("%w: exact Project artifact descriptor is invalid", ErrInvalidSubmission)
	}
	store, err := a.service.Project(projectID)
	if err != nil {
		return artifacts.Payload{}, err
	}
	result, err := store.Read(ctx, descriptor.Ref)
	if err != nil {
		return artifacts.Payload{}, err
	}
	if !sameExactRef(result.Ref, descriptor.Ref) ||
		digestBytes(result.Payload.Data) != descriptor.Digest ||
		result.Payload.MediaType != descriptor.MediaType ||
		int64(len(result.Payload.Data)) != descriptor.SizeBytes {
		return artifacts.Payload{}, fmt.Errorf("%w: exact Project artifact content changed", ErrInvalidSubmission)
	}
	return artifacts.Payload{
		MediaType: result.Payload.MediaType,
		Data:      append([]byte(nil), result.Payload.Data...),
	}, nil
}

type ProjectArtifactAccess struct{ service *artifacts.Service }

func NewProjectArtifactAccess(service *artifacts.Service) (*ProjectArtifactAccess, error) {
	if service == nil {
		return nil, errors.New("Audit Controller Artifact service is required")
	}
	return &ProjectArtifactAccess{service: service}, nil
}

func (a *ProjectArtifactAccess) PutImmutableProject(
	ctx context.Context,
	projectID string,
	target contracts.ArtifactRef,
	payload artifacts.Payload,
) (auditstore.ExactArtifact, error) {
	store, err := a.service.Project(projectID)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	write, err := store.Write(ctx, target, payload, nil)
	if err == nil {
		return auditstore.ExactArtifact{
			Ref: write.Ref, Digest: digestBytes(payload.Data),
			MediaType: write.MediaType, SizeBytes: write.Size,
		}, nil
	}
	if !errors.Is(err, artifacts.ErrArtifactConflict) {
		return auditstore.ExactArtifact{}, err
	}
	// The manifest name is content-derived. A crash after this write but before
	// intent creation is recovered by accepting only byte-identical content.
	current, readErr := store.Read(ctx, target)
	if readErr != nil {
		return auditstore.ExactArtifact{}, readErr
	}
	if current.Payload.MediaType != payload.MediaType || !bytes.Equal(current.Payload.Data, payload.Data) {
		return auditstore.ExactArtifact{}, fmt.Errorf("%w: execution manifest binding collision", ErrInvalidSubmission)
	}
	return auditstore.ExactArtifact{
		Ref: current.Ref, Digest: digestBytes(current.Payload.Data),
		MediaType: current.Payload.MediaType, SizeBytes: int64(len(current.Payload.Data)),
	}, nil
}

func (a *ProjectArtifactAccess) ResolveProjectExact(
	ctx context.Context,
	projectID string,
	descriptor auditstore.ExactArtifact,
) (auditstore.ExactArtifact, error) {
	store, err := a.service.Project(projectID)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	metadata, err := store.Metadata(ctx, descriptor.Ref)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	if metadata.Digest != descriptor.Digest {
		return auditstore.ExactArtifact{}, fmt.Errorf("%w: exact Project artifact digest changed", ErrInvalidSubmission)
	}
	descriptor.Ref = metadata.Ref
	descriptor.MediaType = metadata.MediaType
	descriptor.SizeBytes = metadata.Size
	return descriptor, nil
}

func digestBytes(value []byte) string {
	digest := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(digest[:])
}
