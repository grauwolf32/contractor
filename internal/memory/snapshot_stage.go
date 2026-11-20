package memory

import (
	"context"
	"errors"
	"reflect"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/memorycontract"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

type SnapshotStageRequest struct {
	RunID, StageExecutionID, ClaimID string
	Namespace, Notes                 string
	Output                           contracts.ArtifactRef
}

// SnapshotStage is an intentional export through the existing Memory and Artifact
// owners. Its frozen output binding is the natural replay identity within the Run.
func (s *PostgresStore) SnapshotStage(ctx context.Context, request SnapshotStageRequest, fenceProject func(context.Context, pgx.Tx, string, string) error) (artifacts.Metadata, error) {
	var result artifacts.Metadata
	selection, err := memorycontract.SnapshotOptions(map[string]any{"namespace": request.Namespace, "notes": request.Notes})
	if fenceProject == nil || err != nil || request.Output.Validate() != nil || request.Output.Revision != nil || artifactpolicy.IsPurposeReservedNamespace(request.Output.Namespace) || artifactpolicy.IsModelHiddenBinding(request.Output.Namespace, request.Output.Name) {
		return result, ErrSnapshotInvalid
	}
	ctx, release, err := s.storage.AcquireTransfer(ctx)
	if err != nil {
		return result, err
	}
	defer release()
	err = postgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		result = artifacts.Metadata{}
		store := runstore.NewPostgresTxStore(tx, s.storage)
		owner, err := store.ProjectContext(ctx, request.RunID)
		if err == nil {
			if err := fenceProject(ctx, tx, owner.OwnerID, owner.ProjectID); err != nil {
				return err
			}
		} else if !errors.Is(err, runstore.ErrNotFound) {
			return err
		}
		stage, err := store.LockStageOperation(ctx, request.RunID, request.StageExecutionID, request.ClaimID)
		if err != nil {
			return err
		}
		if stage.State != runstore.StageRunning || len(stage.StageContext.Artifacts) != 0 {
			return ErrSnapshotInvalid
		}
		spec, err := resolved.DecodeResolvedStageSnapshot(stage.StageSpecSnapshot)
		if err != nil || spec.Operation == nil || spec.Operation.Ref != contracts.MemorySnapshotOperation {
			return ErrSnapshotInvalid
		}
		bound := map[string]any{}
		for name, binding := range spec.Operation.Arguments {
			value := binding.Value
			if binding.Source == "parameter" {
				value, err = binding.ResolveParameter(stage.StageContext.Parameters)
				if err != nil {
					return ErrSnapshotInvalid
				}
			}
			bound[name] = value
		}
		if !reflect.DeepEqual(bound, map[string]any{"namespace": request.Namespace, "notes": request.Notes}) {
			return ErrSnapshotInvalid
		}
		slot, exists := spec.Result.Artifacts["snapshot"]
		if !exists || len(spec.Result.Artifacts) != 1 || slot.From == nil || slot.From.Namespace != request.Output.Namespace || slot.From.Name != request.Output.Name {
			return ErrSnapshotInvalid
		}
		service := artifacts.NewService(s.storage.TxRepository(tx))
		scope, err := service.Run(request.RunID)
		if err != nil {
			return err
		}
		// A committed snapshot wins before any mutable note lookup. Mutable or
		// differently shaped collisions are not a snapshot receipt.
		metadata, err := scope.Metadata(ctx, request.Output)
		if err == nil {
			if !metadata.Frozen || metadata.MediaType != "application/json" || metadata.Size > memorycontract.MaximumSnapshotBytes {
				return ErrSnapshotInvalid
			}
			saved, err := scope.ReadExact(ctx, metadata.Exact())
			if err != nil {
				return err
			}
			if err := validateSnapshot(saved.Payload.Data, selection); err != nil {
				return err
			}
			result = metadata
		} else {
			if !errors.Is(err, artifacts.ErrArtifactNotFound) {
				return err
			}
			data, err := buildSnapshot(ctx, scope, selection)
			if err != nil {
				return err
			}
			runScope, _ := artifacts.RunScope(request.RunID)
			written, err := service.CreateFrozen(ctx, runScope, request.Output, artifacts.Payload{MediaType: "application/json", Data: data})
			if err != nil {
				return err
			}
			result, err = scope.Metadata(ctx, written.Ref)
			if err != nil {
				return err
			}
		}
		if s.writeBoundary != nil && s.writeBoundary.beforeCommit != nil {
			s.writeBoundary.beforeCommit(ctx)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		_, err = store.LockStageOperation(ctx, request.RunID, request.StageExecutionID, request.ClaimID)
		return err
	})
	return result, err
}
