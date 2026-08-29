package planner

import (
	"context"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type RunArtifactReader interface {
	Read(context.Context, contracts.ArtifactRef) (artifacts.ReadResult, error)
}

// RunArtifactInspector adapts a RunScope-bound ArtifactStore. The current MVP
// store returns bytes with metadata; Planner immediately discards those bytes.
// A metadata-only store method can replace this adapter without changing the
// Planner contract.
type RunArtifactInspector struct {
	runID string
	store RunArtifactReader
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
	return ArtifactMetadata{MediaType: result.Payload.MediaType}, nil
}

func NewRunArtifactInspector(runID string, store RunArtifactReader) (*RunArtifactInspector, error) {
	if strings.TrimSpace(runID) == "" || store == nil {
		return nil, fmt.Errorf("Run ID and RunScope artifact reader are required")
	}
	return &RunArtifactInspector{runID: runID, store: store}, nil
}

func (i *RunArtifactInspector) Inspect(
	ctx context.Context, runID string, ref contracts.ArtifactRef,
) (ArtifactMetadata, error) {
	if runID != i.runID {
		return ArtifactMetadata{}, fmt.Errorf("artifact inspector is bound to another RunScope")
	}
	if err := ref.ValidateExact(); err != nil {
		return ArtifactMetadata{}, err
	}
	result, err := inspectExact(ctx, i.store, ref)
	if err != nil {
		return ArtifactMetadata{}, err
	}
	return ArtifactMetadata{MediaType: result.Payload.MediaType}, nil
}

func inspectExact(
	ctx context.Context, store RunArtifactReader, ref contracts.ArtifactRef,
) (artifacts.ReadResult, error) {
	result, err := store.Read(ctx, cloneArtifactRef(ref))
	if err != nil {
		return artifacts.ReadResult{}, err
	}
	if result.Ref.Namespace != ref.Namespace || result.Ref.Name != ref.Name ||
		result.Ref.Revision == nil || ref.Revision == nil || *result.Ref.Revision != *ref.Revision {
		return artifacts.ReadResult{}, fmt.Errorf("ArtifactStore resolved a different artifact revision")
	}
	return result, nil
}
