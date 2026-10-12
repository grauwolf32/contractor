package auditcontroller

import (
	"context"
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

// ReadProjectExact reads a complete exact descriptor through the one exact
// read (artifacts.ScopedStore.ReadExact); other content invalidates the
// submission.
func (a *ProjectArtifactAccess) ReadProjectExact(
	ctx context.Context,
	projectID string,
	descriptor auditstore.ExactArtifact,
) (artifacts.Payload, error) {
	if descriptor.Validate() != nil || !descriptor.HasMetadata() {
		return artifacts.Payload{}, fmt.Errorf("%w: exact Project artifact descriptor is invalid", ErrInvalidSubmission)
	}
	store, err := a.service.Project(projectID)
	if err != nil {
		return artifacts.Payload{}, err
	}
	result, err := store.ReadExact(ctx, descriptor)
	if errors.Is(err, contracts.ErrExactArtifactMismatch) {
		return artifacts.Payload{}, fmt.Errorf("%w: exact Project artifact content changed", ErrInvalidSubmission)
	}
	if err != nil {
		return artifacts.Payload{}, err
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

// PutImmutableProject writes a content-named manifest or task package as a
// frozen Audit artifact. A crash after the write but before intent creation
// is recovered by CreateFrozen, which returns the identical binding without
// staging another blob.
func (a *ProjectArtifactAccess) PutImmutableProject(
	ctx context.Context,
	projectID string,
	target contracts.ArtifactRef,
	payload artifacts.Payload,
) (auditstore.ExactArtifact, error) {
	project, err := artifacts.ProjectScope(projectID)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	written, err := a.service.CreateFrozen(ctx, project, target, payload)
	if errors.Is(err, artifacts.ErrAuditArtifactCollision) {
		return auditstore.ExactArtifact{}, fmt.Errorf("%w: immutable Audit artifact binding collision: %v", ErrInvalidSubmission, err)
	}
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	return contracts.ExactContent(written.Ref, written.MediaType, payload.Data), nil
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
	if err := descriptor.Verify(metadata.Exact()); err != nil {
		return auditstore.ExactArtifact{}, fmt.Errorf("%w: exact Project artifact digest changed: %v", ErrInvalidSubmission, err)
	}
	return metadata.Exact(), nil
}
