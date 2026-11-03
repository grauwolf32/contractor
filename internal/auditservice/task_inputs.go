package auditservice

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// retainTaskInput creates one ordinary immutable binding under the current
// owner fence. Metadata-first reconciliation survives source purge and a lost
// fork acknowledgement; no materialization journal or new receipt is needed.
func (s *Service) retainTaskInput(ctx context.Context, audit auditstore.Audit, claim *auditstore.ControllerClaim, revision uint64, source auditcontract.Material, preparedExecution string) (auditcontract.Material, error) {
	var retained auditcontract.Material
	err := postgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		current, err := ownerstore.NewTx(tx, s.storage).LockTaskInputs(ctx, audit.OwnerID, audit.AuditID, claim)
		if err != nil {
			return err
		}
		if claim == nil && current.Revision != revision {
			return auditstore.ErrPrecondition
		}
		write := func(service *artifacts.Service) error {
			var sourceScope artifacts.Scope
			switch source.ScopeKind {
			case "project":
				if source.ScopeID != current.ProjectID {
					return ErrInvalid
				}
				sourceScope, err = artifacts.ProjectScope(source.ScopeID)
			case "audit":
				if source.ScopeID != current.AuditID {
					return ErrInvalid
				}
				sourceScope, err = artifacts.AuditScope(source.ScopeID)
			case "run":
				accepted, readErr := querystore.New(tx, s.storage).CollectedRunInput(ctx, current.AuditID, source.ScopeID)
				if readErr == nil && !accepted && claim != nil && preparedExecution != "" {
					accepted, readErr = querystore.New(tx, s.storage).PreparedRunInput(ctx, current.AuditID, preparedExecution, source.ScopeID)
				}
				if readErr != nil {
					return readErr
				}
				if !accepted {
					return ErrInvalid
				}
				sourceScope, err = artifacts.RunScope(source.ScopeID)
			default:
				return ErrInvalid
			}
			if err != nil {
				return err
			}
			targetScope, _ := artifacts.AuditScope(current.AuditID)
			owned, _ := service.Scoped(targetScope)
			if source.ScopeKind == "audit" {
				metadata, err := owned.Metadata(ctx, source.Ref)
				if err != nil {
					return err
				}
				if !metadata.Frozen || exactTaskMaterial(source).Verify(metadata.Exact()) != nil {
					return ErrInvalid
				}
				retained = source
				return nil
			}
			name, err := auditdomain.MaterialArtifactName(source)
			if err != nil {
				return err
			}
			target := contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(current.AuditID), Name: name}
			metadata, err := owned.Metadata(ctx, target)
			if err == nil {
				wanted := exactTaskMaterial(source)
				wanted.Ref = metadata.Ref
				if !metadata.Frozen || wanted.Verify(metadata.Exact()) != nil {
					return artifacts.ErrAuditArtifactCollision
				}
				retained = scopedTaskMaterial(current.AuditID, metadata)
				return nil
			}
			if !errors.Is(err, artifacts.ErrArtifactNotFound) {
				return err
			}
			selected, _ := service.Scoped(sourceScope)
			metadata, err = selected.Metadata(ctx, source.Ref)
			if err != nil {
				return err
			}
			if exactTaskMaterial(source).Verify(metadata.Exact()) != nil {
				return ErrInvalid
			}
			kind := artifacts.LineageAuditInput
			if source.ScopeKind == "run" {
				kind = artifacts.LineageAuditOutput
			}
			fork, err := service.Fork(ctx, artifacts.Fork{Kind: kind, Source: sourceScope, SourceRef: source.Ref, Target: targetScope, TargetRef: target, Frozen: true, Replay: true})
			if err != nil {
				return err
			}
			metadata, err = owned.Metadata(ctx, fork.TargetRef)
			if err != nil {
				return err
			}
			retained = scopedTaskMaterial(current.AuditID, metadata)
			return nil
		}
		if claim != nil {
			return hookstore.NewClaimedArtifacts(tx, s.storage).WithArtifactWrite(ctx, *claim, current.ProjectID, auditdomain.ArtifactNamespace(current.AuditID), write)
		}
		return write(artifacts.NewService(s.storage.TxRepository(tx)))
	})
	if err != nil {
		return auditcontract.Material{}, err
	}
	return retained, nil
}

func exactTaskMaterial(material auditcontract.Material) contracts.ExactArtifact {
	return contracts.ExactArtifact{Ref: material.Ref, Digest: material.Digest, MediaType: material.MediaType, SizeBytes: material.SizeBytes}
}

func scopedTaskMaterial(auditID string, metadata artifacts.Metadata) auditcontract.Material {
	return auditcontract.Material{ScopeKind: "audit", ScopeID: auditID, Ref: metadata.Ref.Clone(), Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}
}
