package scheduler

import (
	"context"
	"errors"
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
	result, err := store.Metadata(ctx, ref)
	if err != nil {
		return ResolvedArtifact{}, err
	}
	if err := result.Ref.ValidateExact(); err != nil {
		return ResolvedArtifact{}, fmt.Errorf("ArtifactStore returned an inexact ref: %w", err)
	}
	return ResolvedArtifact{Ref: result.Ref, MediaType: result.MediaType}, nil
}

// outputsNamespace holds a Run's Workflow outputs, which only the Scheduler
// binds (ART-R89).
const outputsNamespace = "outputs"

// bindOutput creates the Workflow output slot of runID at the exact StageResult
// revision source, a metadata fork inside the RunScope with lineage
// output_bind (ART-R90). Pass a Service bound to the transaction that accepts
// the StageResult, so the binding commits with it. A Run creates each output
// binding once: one already forked from the same exact source revision is a
// replay and is returned unchanged, and one of any other revision is never
// replaced, which errInvalidRunState reports: no replay of the acceptance
// changes the existing binding.
func bindOutput(
	ctx context.Context, service *artifacts.Service, runID, slot string, source contracts.ArtifactRef,
) (artifacts.ForkResult, error) {
	run, err := artifacts.RunScope(runID)
	if err != nil {
		return artifacts.ForkResult{}, err
	}
	if source.Revision == nil {
		return artifacts.ForkResult{}, artifacts.ErrExactRevisionRequired
	}
	bound, err := service.Fork(ctx, artifacts.Fork{
		Kind: artifacts.LineageOutputBind, Source: run, SourceRef: source,
		Target: run, TargetRef: contracts.ArtifactRef{Namespace: outputsNamespace, Name: slot},
		Replay: true,
	})
	// A conflict means the target exists, which no replay changes; a
	// transient database failure is no conflict and stays an ordinary error
	// for finalization recovery to retry.
	if errors.Is(err, artifacts.ErrArtifactConflict) {
		return artifacts.ForkResult{}, fmt.Errorf("%w: %w", errInvalidRunState, err)
	}
	return bound, err
}

// freezeOutputs freezes the output bindings of runID as its Run becomes
// terminal (ART-R92, ART-R94); a Run without outputs freezes nothing.
func freezeOutputs(ctx context.Context, service *artifacts.Service, runID string) error {
	run, err := artifacts.RunScope(runID)
	if err != nil {
		return err
	}
	return service.Freeze(ctx, run, outputsNamespace)
}

// publishRunOutput creates the Project binding outputs/<slot> of projectID
// from the exact frozen output source of runID (ART-R99). It only creates: an
// existing Project binding is a conflict, never an implicit replacement.
func publishRunOutput(
	ctx context.Context, service *artifacts.Service, runID, projectID, slot string, source contracts.ArtifactRef,
) (artifacts.ForkResult, error) {
	run, err := artifacts.RunScope(runID)
	if err != nil {
		return artifacts.ForkResult{}, err
	}
	project, err := artifacts.ProjectScope(projectID)
	if err != nil {
		return artifacts.ForkResult{}, err
	}
	if source.Namespace != outputsNamespace || source.Name != slot {
		return artifacts.ForkResult{}, artifacts.ErrInvalidName
	}
	if source.Revision == nil {
		return artifacts.ForkResult{}, artifacts.ErrExactRevisionRequired
	}
	return service.Fork(ctx, artifacts.Fork{
		Kind: artifacts.LineageProjectOutputPublish, Source: run, SourceRef: source, SourceFrozen: true,
		Target: project, TargetRef: contracts.ArtifactRef{Namespace: outputsNamespace, Name: slot},
	})
}
