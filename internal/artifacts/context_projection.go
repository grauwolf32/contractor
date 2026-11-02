package artifacts

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

// CreateContextProjection is transaction-only: creation and exact lineage
// protection must commit together. The trusted caller validates and authorizes
// derived payload content. This method never copies the unfiltered source.
func (r *PostgresTxRepository) CreateContextProjection(ctx context.Context, source Scope, sourceRef ArtifactRef, target Scope, targetRef ArtifactRef, payload Payload) (WriteResult, bool, error) {
	if validateScope(source) != nil || validateScope(target) != nil || source.kind != ScopeProject || target.kind != ScopeRun {
		return WriteResult{}, false, ErrInvalidScope
	}
	if err := validateRef(sourceRef); err != nil {
		return WriteResult{}, false, err
	}
	if sourceRef.Revision == nil {
		return WriteResult{}, false, ErrExactRevisionRequired
	}
	if err := validateCreate(target, targetRef, payload, true); err != nil {
		return WriteResult{}, false, err
	}
	var member bool
	if err := r.tx.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM workflow_runs WHERE run_id=$1 AND project_id=$2)`, target.id, source.id).Scan(&member); err != nil {
		return WriteResult{}, false, err
	}
	if !member {
		return WriteResult{}, false, ErrArtifactNotFound
	}
	if err := r.RequireExact(ctx, source, sourceRef); err != nil {
		return WriteResult{}, false, err
	}
	current, err := r.Metadata(ctx, target, targetRef)
	if err == nil {
		if !current.Frozen || current.Digest != contentdigest.Bytes(payload.Data) || current.MediaType != payload.MediaType {
			return WriteResult{}, false, contracts.ErrIdempotencyConflict
		}
		var matches bool
		if err := r.tx.QueryRow(ctx, contextProjectionReplaySQL, target.kind, target.id, targetRef.Namespace, targetRef.Name, current.Ref.Revision, source.kind, source.id, sourceRef.Namespace, sourceRef.Name, sourceRef.Revision, LineageContextProjection).Scan(&matches); err != nil {
			return WriteResult{}, false, err
		}
		if !matches {
			return WriteResult{}, false, contracts.ErrIdempotencyConflict
		}
		return WriteResult{Ref: current.Ref, MediaType: current.MediaType, Size: current.Size, BindingCreatedAt: current.BindingCreatedAt, RevisionCreatedAt: current.CreatedAt}, true, nil
	}
	if !errors.Is(err, ErrArtifactNotFound) {
		return WriteResult{}, false, err
	}
	result, err := NewService(r).CreateFrozen(ctx, target, targetRef, payload)
	if err != nil {
		return WriteResult{}, false, err
	}
	inserted, err := r.tx.Exec(ctx, createContextProjectionLineageSQL, target.kind, target.id, targetRef.Namespace, targetRef.Name, result.Ref.Revision, source.kind, source.id, sourceRef.Namespace, sourceRef.Name, sourceRef.Revision, LineageContextProjection)
	if err != nil {
		return WriteResult{}, false, storeError(fmt.Sprintf("retain context projection %s/%s", targetRef.Namespace, targetRef.Name), err, nil)
	}
	if inserted.RowsAffected() == 0 {
		// CreateFrozen can return a concurrent winner. Reconcile its lineage
		// in a fresh snapshot without swallowing a different source identity.
		var matches bool
		if err := r.tx.QueryRow(ctx, contextProjectionReplaySQL, target.kind, target.id,
			targetRef.Namespace, targetRef.Name, result.Ref.Revision, source.kind, source.id,
			sourceRef.Namespace, sourceRef.Name, sourceRef.Revision, LineageContextProjection).Scan(&matches); err != nil {
			return WriteResult{}, false, err
		}
		if !matches {
			return WriteResult{}, false, contracts.ErrIdempotencyConflict
		}
		return result, true, nil
	}
	return result, false, nil
}
