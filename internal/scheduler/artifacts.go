package scheduler

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// ArtifactServiceResolver resolves current RunScope bindings outside any
// Scheduler transaction and returns only the exact ref plus metadata.
type ArtifactServiceResolver struct {
	service *artifacts.Service
}

func NewArtifactServiceResolver(service *artifacts.Service) (*ArtifactServiceResolver, error) {
	if service == nil {
		return nil, fmt.Errorf("ArtifactStore service is required")
	}
	return &ArtifactServiceResolver{service: service}, nil
}

func (r *ArtifactServiceResolver) Resolve(
	ctx context.Context,
	runID string,
	ref contracts.ArtifactRef,
) (ResolvedArtifact, error) {
	store, err := r.service.Run(runID)
	if err != nil {
		return ResolvedArtifact{}, err
	}
	result, err := store.Read(ctx, ref)
	if err != nil {
		return ResolvedArtifact{}, err
	}
	if err := result.Ref.ValidateExact(); err != nil {
		return ResolvedArtifact{}, fmt.Errorf("ArtifactStore returned an inexact ref: %w", err)
	}
	return ResolvedArtifact{Ref: result.Ref, MediaType: result.Payload.MediaType}, nil
}
