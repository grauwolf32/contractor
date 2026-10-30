package planner

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type RunArtifactMetadataReader interface {
	Metadata(context.Context, contracts.ArtifactRef) (artifacts.Metadata, error)
}

// ArtifactServiceInspector resolves the RunScope supplied by each invocation
// and is suitable for a process-wide PlannerFactory registry.
type ArtifactServiceInspector struct {
	service *artifacts.Service
}

func NewArtifactServiceInspector(service *artifacts.Service) (*ArtifactServiceInspector, error) {
	if service == nil {
		return nil, fmt.Errorf("ArtifactStore service is required")
	}
	return &ArtifactServiceInspector{service: service}, nil
}

func (i *ArtifactServiceInspector) Inspect(
	ctx context.Context, runID string, ref contracts.ArtifactRef,
) (ArtifactMetadata, error) {
	store, err := i.service.Run(runID)
	if err != nil {
		return ArtifactMetadata{}, err
	}
	result, err := inspectExact(ctx, store, ref)
	if err != nil {
		return ArtifactMetadata{}, err
	}
	return ArtifactMetadata{MediaType: result.MediaType}, nil
}

// inspectExact reads the metadata of the exact revision ref names. Its errors
// keep the classification of artifacts.IsExactReferenceRejection: a malformed
// ref is a contracts validation error, and metadata that describes another
// revision is ErrArtifactIntegrity, which no repeated read repairs.
func inspectExact(
	ctx context.Context, store RunArtifactMetadataReader, ref contracts.ArtifactRef,
) (artifacts.Metadata, error) {
	if err := ref.ValidateExact(); err != nil {
		return artifacts.Metadata{}, err
	}
	result, err := store.Metadata(ctx, ref.Clone())
	if err != nil {
		return artifacts.Metadata{}, err
	}
	if !result.Ref.SameExact(ref) {
		return artifacts.Metadata{}, fmt.Errorf(
			"%w: ArtifactStore resolved a different artifact revision", artifacts.ErrArtifactIntegrity,
		)
	}
	return result, nil
}
