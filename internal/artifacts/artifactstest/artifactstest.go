// Package artifactstest sets up, for tests, the Artifact state that the
// purpose-specific Server flows of other packages create: Run input forks,
// Workflow output bindings, finding proposals and Audit material. Each helper
// uses the trusted primitive and the Namespace of the flow that owns the
// operation in production.
package artifactstest

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// ForkInput forks source of the UserScope of userID into inputs/<slot> of
// runID, as standalone Run creation does.
func ForkInput(
	ctx context.Context, service *artifacts.Service, userID string, source contracts.ArtifactRef, runID, slot string,
) (artifacts.ForkResult, error) {
	user, err := artifacts.UserScope(userID)
	if err != nil {
		return artifacts.ForkResult{}, err
	}
	return forkInput(ctx, service, user, source, runID, slot)
}

// ForkProjectInput forks source of the ProjectScope of projectID into
// inputs/<slot> of runID, as Project Run creation does.
func ForkProjectInput(
	ctx context.Context, service *artifacts.Service, projectID string, source contracts.ArtifactRef, runID, slot string,
) (artifacts.ForkResult, error) {
	project, err := artifacts.ProjectScope(projectID)
	if err != nil {
		return artifacts.ForkResult{}, err
	}
	return forkInput(ctx, service, project, source, runID, slot)
}

func forkInput(
	ctx context.Context, service *artifacts.Service, scope artifacts.Scope, source contracts.ArtifactRef, runID, slot string,
) (artifacts.ForkResult, error) {
	run, err := artifacts.RunScope(runID)
	if err != nil {
		return artifacts.ForkResult{}, err
	}
	return service.Fork(ctx, artifacts.Fork{
		Kind: artifacts.LineageInputFork, Source: scope, SourceRef: source,
		Target: run, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: slot},
	})
}

// BindOutput binds the exact revision source of runID to outputs/<slot>, as
// the Scheduler does when it accepts a successful StageResult.
func BindOutput(
	ctx context.Context, service *artifacts.Service, runID, slot string, source contracts.ArtifactRef,
) (artifacts.ForkResult, error) {
	run, err := artifacts.RunScope(runID)
	if err != nil {
		return artifacts.ForkResult{}, err
	}
	return service.Fork(ctx, artifacts.Fork{
		Kind: artifacts.LineageOutputBind, Source: run, SourceRef: source,
		Target: run, TargetRef: contracts.ArtifactRef{Namespace: "outputs", Name: slot},
	})
}

// FreezeOutputs freezes the outputs of runID, as the Scheduler does when the
// Run becomes terminal.
func FreezeOutputs(ctx context.Context, service *artifacts.Service, runID string) error {
	run, err := artifacts.RunScope(runID)
	if err != nil {
		return err
	}
	return service.Freeze(ctx, run, "outputs")
}

// WriteFindingProposal writes payload as the finding proposal name of runID,
// as finding intake does.
func WriteFindingProposal(
	ctx context.Context, service *artifacts.Service, runID, name string, payload artifacts.Payload,
) (artifacts.WriteResult, error) {
	run, err := artifacts.RunScope(runID)
	if err != nil {
		return artifacts.WriteResult{}, err
	}
	return service.Create(ctx, run,
		contracts.ArtifactRef{Namespace: artifactpolicy.FindingProposalNamespace, Name: name}, payload)
}

// WriteAuditArtifact writes payload as the frozen Audit binding target of
// projectID, as the Audit packages do.
func WriteAuditArtifact(
	ctx context.Context, service *artifacts.Service, projectID string, target contracts.ArtifactRef, payload artifacts.Payload,
) (artifacts.WriteResult, error) {
	project, err := artifacts.ProjectScope(projectID)
	if err != nil {
		return artifacts.WriteResult{}, err
	}
	return service.CreateFrozen(ctx, project, target, payload)
}
