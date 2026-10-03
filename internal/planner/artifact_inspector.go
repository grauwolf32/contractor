package planner

import (
	"context"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type RunArtifactMetadataReader interface {
	Metadata(context.Context, contracts.ArtifactRef) (artifacts.Metadata, error)
}

// RunArtifactInspector adapts a RunScope-bound metadata store.
type RunArtifactInspector struct {
	runID string
	store RunArtifactMetadataReader
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

func NewRunArtifactInspector(runID string, store RunArtifactMetadataReader) (*RunArtifactInspector, error) {
	if strings.TrimSpace(runID) == "" || store == nil {
		return nil, fmt.Errorf("Run ID and RunScope artifact metadata reader are required")
	}
	return &RunArtifactInspector{runID: runID, store: store}, nil
}

func (i *RunArtifactInspector) Inspect(
	ctx context.Context, runID string, ref contracts.ArtifactRef,
) (ArtifactMetadata, error) {
	if runID != i.runID {
		return ArtifactMetadata{}, fmt.Errorf("artifact inspector is bound to another RunScope")
	}
	result, err := inspectExact(ctx, i.store, ref)
	if err != nil {
		return ArtifactMetadata{}, err
	}
	return ArtifactMetadata{MediaType: result.MediaType}, nil
}

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
	if result.Ref.Namespace != ref.Namespace || result.Ref.Name != ref.Name ||
		result.Ref.Revision == nil || ref.Revision == nil || *result.Ref.Revision != *ref.Revision {
		return artifacts.Metadata{}, fmt.Errorf("ArtifactStore resolved a different artifact revision")
	}
	return result, nil
}
