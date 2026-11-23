// Package artifactcopy binds an exact Stage input to ArtifactStore's existing fork.
package artifactcopy

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/operations"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Module struct {
	pool    *pgxpool.Pool
	storage *artifactpostgres.Storage
}

func New(pool *pgxpool.Pool, storage *artifactpostgres.Storage) *Module {
	return &Module{pool: pool, storage: storage}
}

func (*Module) Ref() string { return contracts.ArtifactCopyOperation }

func invalid() error {
	return &operations.Failure{Code: "invalid_request", Message: "Artifact copy requires its exact source and declared copied output"}
}

func (m *Module) Run(ctx context.Context, call operations.Call) (stageresult.StageContentResult, error) {
	empty := stageresult.StageContentResult{}
	if len(call.Arguments) != 0 || len(call.Inputs) != 1 || len(call.Outputs) != 1 || call.Inputs["source"].ValidateExact() != nil || artifactpolicy.IsModelHiddenBinding(call.Inputs["source"].Namespace, call.Inputs["source"].Name) {
		return empty, invalid()
	}
	output, exists := call.Outputs["copied"]
	if !exists || output.Validate() != nil || output.Revision != nil || artifactpolicy.IsModelHiddenBinding(output.Namespace, output.Name) {
		return empty, invalid()
	}
	if m.pool == nil || m.storage == nil {
		return empty, &operations.Failure{Code: "server_operation_unavailable", Message: "Artifact copy storage is unavailable"}
	}
	admitted, release, err := m.storage.AcquireTransfer(ctx)
	if err != nil {
		return empty, err
	}
	defer release()
	var copied contracts.ArtifactRef
	err = postgres.InTx(admitted, m.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		store := runstore.NewPostgresTxStore(tx, m.storage)
		owner, err := store.ProjectContext(admitted, call.RunID)
		if err == nil {
			if _, err = projectstore.NewPostgresStore(tx).LockActiveProject(admitted, owner.OwnerID, owner.ProjectID); err != nil {
				return err
			}
		} else if !errors.Is(err, runstore.ErrNotFound) {
			return err
		}
		stage, err := store.LockStageOperation(admitted, call.RunID, call.StageExecutionID, call.ClaimID)
		if err != nil {
			return err
		}
		spec, err := resolved.DecodeResolvedStageSnapshot(stage.StageSpecSnapshot)
		if err != nil || stage.State != runstore.StageRunning || spec.Operation == nil || spec.Operation.Ref != contracts.ArtifactCopyOperation || len(spec.Operation.Arguments) != 0 {
			return invalid()
		}
		input, exists := stage.StageContext.Artifacts["source"]
		slot, declared := spec.Result.Artifacts["copied"]
		if !exists || !input.Required || input.Artifact == nil || !input.Artifact.SameExact(call.Inputs["source"]) || !declared || slot.From == nil || slot.From.Namespace != output.Namespace || slot.From.Name != output.Name {
			return invalid()
		}
		scope, err := artifacts.RunScope(call.RunID)
		if err != nil {
			return err
		}
		fork, err := artifacts.NewService(m.storage.TxRepository(tx)).Fork(admitted, artifacts.Fork{
			Source: scope, SourceRef: call.Inputs["source"], Target: scope, TargetRef: output,
			Kind: artifacts.LineageInputFork, Replay: true,
		})
		if err != nil {
			return err
		}
		copied = fork.TargetRef
		if err := admitted.Err(); err != nil {
			return err
		}
		_, err = store.LockStageOperation(admitted, call.RunID, call.StageExecutionID, call.ClaimID)
		return err
	})
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, runstore.ErrConflict) {
			return empty, err
		}
		if kind, ok := apierror.Of(err); ok {
			k := kind.ErrorKind()
			return empty, &operations.Failure{Code: k.Code(), Message: k.Message(), Retryable: k.Retryable()}
		}
		return empty, err
	}
	return stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded,
		Summary: "Copied the assigned exact artifact.", Artifacts: map[string]contracts.ArtifactRef{"copied": copied},
	}, nil
}
