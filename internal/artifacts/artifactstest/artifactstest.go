// Package artifactstest sets up, for tests, the Artifact state that the
// purpose-specific Server flows of other packages create: Run input forks,
// Workflow output bindings, finding proposals and Audit material. Each helper
// uses the trusted primitive and the Namespace of the flow that owns the
// operation in production.
package artifactstest

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
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
// the Scheduler does when it accepts a successful StageResult. It only
// creates: unlike the Scheduler, which replays a binding of the same source
// (ART-R90), it reports every existing binding as a conflict.
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

// WriteRunArtifact writes payload to target of runID as a Worker's ordinary
// write does, creating the binding or advancing its current revision, and
// returns the exact ref of the revision it wrote. An exact ref that a
// StageExecution, a receipt or an Audit records must name a stored revision.
func WriteRunArtifact(
	ctx context.Context, service *artifacts.Service, runID string, target contracts.ArtifactRef, payload artifacts.Payload,
) (contracts.ArtifactRef, error) {
	run, err := service.Run(runID)
	if err != nil {
		return contracts.ArtifactRef{}, err
	}
	var expected *string
	current, err := run.Metadata(ctx, contracts.ArtifactRef{Namespace: target.Namespace, Name: target.Name})
	switch {
	case err == nil:
		expected = current.Ref.Revision
	case !errors.Is(err, artifacts.ErrArtifactNotFound):
		return contracts.ArtifactRef{}, err
	}
	written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: target.Namespace, Name: target.Name}, payload, expected)
	if err != nil {
		return contracts.ArtifactRef{}, err
	}
	return written.Ref, nil
}

// LoseRunRevision removes the exact revision ref of runID's RunScope while
// holders may still name it, which in production only the controlled purge
// of that scope does (DEL-R65): fixtures use it to model a cited revision
// that is no longer stored. ref must not be the binding's current revision.
func LoseRunRevision(ctx context.Context, pool *pgxpool.Pool, runID string, ref contracts.ArtifactRef) error {
	if ref.Revision == nil {
		return errors.New("lose a Run revision: the ref names no exact revision")
	}
	return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SELECT set_config('contractor.lifecycle_purge', 'run', true)`); err != nil {
			return err
		}
		removed, err := tx.Exec(ctx, `DELETE FROM artifact_binding_revisions
WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = $2 AND name = $3 AND revision = $4`,
			runID, ref.Namespace, ref.Name, *ref.Revision)
		if err == nil && removed.RowsAffected() != 1 {
			err = errors.New("lose a Run revision: the revision does not exist")
		}
		return err
	})
}
