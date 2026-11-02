package openapiwork

import (
	"bytes"
	"context"
	"errors"
	"reflect"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/operations"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/scanplan"
	"github.com/jackc/pgx/v5"
)

// withOwner acquires the existing Project/Run/Stage fences in their established
// order. Parsing is outside these transactions; each publication owns one bounded
// Artifact write, with a final live-claim check before commit.
func (m *Module) withOwner(ctx context.Context, call operations.Call, fn func(context.Context, *artifacts.Service) error) error {
	admitted, release, err := artifacts.AcquireTransfer(ctx)
	if err != nil {
		return err
	}
	defer release()
	return postgres.InTx(admitted, m.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		store := runstore.NewPostgresTxStore(tx)
		owner, err := store.ProjectContext(admitted, call.RunID)
		if err != nil {
			return err
		}
		if _, err = projectstore.NewPostgresStore(tx).LockActiveAuditProject(admitted, owner.OwnerID, owner.ProjectID); err != nil {
			return err
		}
		stage, err := store.LockStageOperation(admitted, call.RunID, call.StageExecutionID, call.ClaimID)
		if err != nil {
			return err
		}
		if stage.State != runstore.StageRunning {
			return runstore.ErrConflict
		}
		spec, err := resolved.DecodeResolvedStageSnapshot(stage.StageSpecSnapshot)
		if err != nil || spec.Operation == nil || spec.Operation.Ref != contracts.OpenAPIPrepareWorkOperation {
			return invalid()
		}
		bound := map[string]any{}
		for name, binding := range spec.Operation.Arguments {
			if binding.Source == "literal" {
				bound[name] = binding.Value
			} else {
				value, err := binding.ResolveParameter(stage.StageContext.Parameters)
				if err != nil {
					return invalid()
				}
				bound[name] = value
			}
		}
		if !reflect.DeepEqual(bound, call.Arguments) || len(call.Outputs) != len(spec.Result.Artifacts) {
			return invalid()
		}
		for name, target := range call.Outputs {
			slot, exists := spec.Result.Artifacts[name]
			if !exists || slot.From == nil || target.Namespace != slot.From.Namespace || target.Name != slot.From.Name || target.Revision != nil {
				return invalid()
			}
		}
		selected := 0
		for name, input := range stage.StageContext.Artifacts {
			if input.Artifact == nil {
				continue
			}
			selected++
			actual, exists := call.Inputs[name]
			if !exists || !input.Artifact.SameExact(actual) {
				return invalid()
			}
		}
		if selected != len(call.Inputs) {
			return invalid()
		}
		if err := fn(admitted, artifacts.NewService(artifacts.NewPostgresTxRepository(tx))); err != nil {
			return err
		}
		if err := admitted.Err(); err != nil {
			return err
		}
		_, err = store.LockStageOperation(admitted, call.RunID, call.StageExecutionID, call.ClaimID)
		return err
	})
}

func (m *Module) readInputs(ctx context.Context, call operations.Call, options auditcontract.OpenAPIWorkArguments) (auditdomain.PreparedOpenAPIInput, error) {
	input := auditdomain.PreparedOpenAPIInput{Options: options}
	err := m.withOwner(ctx, call, func(ctx context.Context, service *artifacts.Service) error {
		store, err := service.Run(call.RunID)
		if err != nil {
			return err
		}
		for name, ref := range call.Inputs {
			metadata, err := store.Metadata(ctx, ref)
			if err != nil {
				return err
			}
			limit := scanplan.MaxSourceBytes
			if name == "settings" {
				limit = scanplan.MaxOptionsBytes
				if metadata.MediaType != "application/json" {
					return invalid()
				}
			}
			if name == "openapi" && metadata.MediaType == auditdomain.PackageMediaType {
				limit = auditdomain.MaximumArchiveBytes
			}
			if metadata.Size < 0 || metadata.Size > int64(limit) {
				return invalid()
			}
			payload, err := store.ReadExact(ctx, metadata.Exact())
			if err != nil {
				return err
			}
			material := descriptor(call.RunID, metadata)
			if name == "settings" {
				input.Settings = &material
				input.SettingsData = payload.Payload.Data
			} else {
				input.Source = material
				input.Data = payload.Payload.Data
			}
		}
		return nil
	})
	return input, err
}

func (m *Module) publish(ctx context.Context, call operations.Call, target contracts.ArtifactRef, data []byte) (auditcontract.Material, error) {
	var result auditcontract.Material
	err := m.withOwner(ctx, call, func(ctx context.Context, service *artifacts.Service) error {
		store, err := service.Run(call.RunID)
		if err != nil {
			return err
		}
		metadata, err := store.Metadata(ctx, target)
		if err == nil {
			if metadata.MediaType != "application/json" || metadata.Size != int64(len(data)) {
				return artifacts.ErrArtifactConflict
			}
			prior, err := store.ReadExact(ctx, metadata.Exact())
			if err != nil {
				return err
			}
			if !bytes.Equal(prior.Payload.Data, data) {
				return artifacts.ErrArtifactConflict
			}
			result = descriptor(call.RunID, metadata)
			return nil
		}
		if !errors.Is(err, artifacts.ErrArtifactNotFound) {
			return err
		}
		scope, _ := artifacts.RunScope(call.RunID)
		written, err := service.Create(ctx, scope, target, artifacts.Payload{MediaType: "application/json", Data: data})
		if err != nil {
			return err
		}
		metadata, err = store.Metadata(ctx, written.Ref)
		if err != nil {
			return err
		}
		result = descriptor(call.RunID, metadata)
		return nil
	})
	return result, err
}
