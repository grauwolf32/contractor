package scheduler

import (
	"bytes"
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
)

// ArtifactServiceResolver resolves current RunScope bindings outside any
// Scheduler transaction, and publishes source-keyed summary artifacts.
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

var errSummaryArtifactConflict = errors.New("saved summary differs from its source completion")

// SaveSummary uses the source execution as a natural publication key. A lost
// acknowledgement is reconciled by exact bytes, never by appending a revision.
func (r *ArtifactServiceResolver) SaveSummary(
	ctx context.Context, runID, namespace, sourceStageID, summary string,
) (contracts.ArtifactRef, error) {
	store, err := r.service.Run(runID)
	if err != nil {
		return contracts.ArtifactRef{}, err
	}
	ref := contracts.ArtifactRef{Namespace: namespace, Name: stageresult.SummaryArtifactPrefix + sourceStageID}
	read := func() (contracts.ArtifactRef, error) {
		value, err := store.Read(ctx, ref)
		if err != nil {
			return contracts.ArtifactRef{}, err
		}
		if value.Payload.MediaType != "text/plain" || !bytes.Equal(value.Payload.Data, []byte(summary)) {
			return contracts.ArtifactRef{}, errSummaryArtifactConflict
		}
		return value.Ref.Clone(), nil
	}
	if existing, err := read(); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		return existing, err
	}
	written, err := store.Write(ctx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte(summary)}, nil)
	if err == nil {
		return written.Ref.Clone(), nil
	}
	if ctx.Err() == nil {
		if existing, readErr := read(); readErr == nil || errors.Is(readErr, errSummaryArtifactConflict) {
			return existing, readErr
		}
	}
	return contracts.ArtifactRef{}, err
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
