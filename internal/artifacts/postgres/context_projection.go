package postgres

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

// CreateContextProjection is transaction-only: creation and exact lineage
// protection must commit together. The trusted caller validates and authorizes
// derived payload content. This method never copies the unfiltered source.
func (r *TxRepository) CreateContextProjection(ctx context.Context, source artifacts.Scope, sourceRef artifacts.ArtifactRef, target artifacts.Scope, targetRef artifacts.ArtifactRef, payload artifacts.Payload) (artifacts.WriteResult, bool, error) {
	if artifacts.ValidateScope(source) != nil || artifacts.ValidateScope(target) != nil || source.Kind() != artifacts.ScopeProject || target.Kind() != artifacts.ScopeRun {
		return artifacts.WriteResult{}, false, artifacts.ErrInvalidScope
	}
	if err := artifacts.ValidateRef(sourceRef); err != nil {
		return artifacts.WriteResult{}, false, err
	}
	if sourceRef.Revision == nil {
		return artifacts.WriteResult{}, false, artifacts.ErrExactRevisionRequired
	}
	if err := artifacts.ValidateCreate(target, targetRef, payload, true); err != nil {
		return artifacts.WriteResult{}, false, err
	}
	var member bool
	if err := r.tx.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM workflow_runs WHERE run_id=$1 AND project_id=$2)`, target.ID(), source.ID()).Scan(&member); err != nil {
		return artifacts.WriteResult{}, false, err
	}
	if !member {
		return artifacts.WriteResult{}, false, artifacts.ErrArtifactNotFound
	}
	if err := r.RequireExact(ctx, source, sourceRef); err != nil {
		return artifacts.WriteResult{}, false, err
	}
	current, err := r.Metadata(ctx, target, targetRef)
	if err == nil {
		if !current.Frozen || current.Digest != contentdigest.Bytes(payload.Data) || current.MediaType != payload.MediaType {
			return artifacts.WriteResult{}, false, contracts.ErrIdempotencyConflict
		}
		var matches bool
		if err := r.tx.QueryRow(ctx, contextProjectionReplaySQL, target.Kind(), target.ID(), targetRef.Namespace, targetRef.Name, current.Ref.Revision, source.Kind(), source.ID(), sourceRef.Namespace, sourceRef.Name, sourceRef.Revision, artifacts.LineageContextProjection).Scan(&matches); err != nil {
			return artifacts.WriteResult{}, false, err
		}
		if !matches {
			return artifacts.WriteResult{}, false, contracts.ErrIdempotencyConflict
		}
		return artifacts.WriteResult{Ref: current.Ref, MediaType: current.MediaType, Size: current.Size, BindingCreatedAt: current.BindingCreatedAt, RevisionCreatedAt: current.CreatedAt}, true, nil
	}
	if !errors.Is(err, artifacts.ErrArtifactNotFound) {
		return artifacts.WriteResult{}, false, err
	}
	result, err := artifacts.NewService(r).CreateFrozen(ctx, target, targetRef, payload)
	if err != nil {
		return artifacts.WriteResult{}, false, err
	}
	inserted, err := r.tx.Exec(ctx, createContextProjectionLineageSQL, target.Kind(), target.ID(), targetRef.Namespace, targetRef.Name, result.Ref.Revision, source.Kind(), source.ID(), sourceRef.Namespace, sourceRef.Name, sourceRef.Revision, artifacts.LineageContextProjection)
	if err != nil {
		return artifacts.WriteResult{}, false, storeError(fmt.Sprintf("retain context projection %s/%s", targetRef.Namespace, targetRef.Name), err, nil)
	}
	if inserted.RowsAffected() == 0 {
		// CreateFrozen can return a concurrent winner. Reconcile its lineage
		// in a fresh snapshot without swallowing a different source identity.
		var matches bool
		if err := r.tx.QueryRow(ctx, contextProjectionReplaySQL, target.Kind(), target.ID(),
			targetRef.Namespace, targetRef.Name, result.Ref.Revision, source.Kind(), source.ID(),
			sourceRef.Namespace, sourceRef.Name, sourceRef.Revision, artifacts.LineageContextProjection).Scan(&matches); err != nil {
			return artifacts.WriteResult{}, false, err
		}
		if !matches {
			return artifacts.WriteResult{}, false, contracts.ErrIdempotencyConflict
		}
		return result, true, nil
	}
	return result, false, nil
}
