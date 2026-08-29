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
	result, err := i.store.Read(ctx, cloneArtifactRef(ref))
	if err != nil {
		return ArtifactMetadata{}, err
	}
	if result.Ref.Namespace != ref.Namespace || result.Ref.Name != ref.Name ||
		result.Ref.Revision == nil || ref.Revision == nil || *result.Ref.Revision != *ref.Revision {
		return ArtifactMetadata{}, fmt.Errorf("ArtifactStore resolved a different artifact revision")
	}
	return ArtifactMetadata{MediaType: result.Payload.MediaType}, nil
}
