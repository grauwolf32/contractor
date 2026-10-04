package artifacts

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/clone"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (r *PostgresRepository) Write(
	ctx context.Context,
	scope Scope,
	target ArtifactRef,
	payload Payload,
	expectedRevision *string,
) (WriteResult, error) {
	if err := validateScope(scope); err != nil {
		return WriteResult{}, err
	}
	if err := validateRef(target); err != nil {
		return WriteResult{}, err
	}
	if target.Revision != nil {
		return WriteResult{}, ErrVersionedWriteTarget
	}
	if expectedRevision != nil {
		if err := validateRevision(*expectedRevision); err != nil {
			return WriteResult{}, err
		}
	}
	if scope.kind == ScopeRun && target.Namespace == "outputs" {
		return WriteResult{}, ErrReservedNamespace
	}
	if err := validatePayload(payload); err != nil {
		return WriteResult{}, err
	}
	revision, err := r.newID("rev_")
	if err != nil {
		return WriteResult{}, err
	}
	versionID, err := r.newID("version_")
	if err != nil {
		return WriteResult{}, err
	}
	ctx, releaseTransfer, err := AcquireTransfer(ctx)
	if err != nil {
		return WriteResult{}, err
	}
	defer releaseTransfer()
	blob, err := prepareBlob(ctx, payload)
	if err != nil {
		return WriteResult{}, err
	}
	discardFreshCandidate := func() {
		// A preprepared file may have been published by an earlier use of the
		// same Payload; only this Write's fresh candidate is ours to remove.
		if payload.prepared == nil {
			discardUnpublishedBlob(ctx, blob)
		}
	}
	digest := blob.Digest
	reusableKey, err := reusableBlobKey(ctx, r.db, blob)
	if err != nil {
		discardFreshCandidate()
		return WriteResult{}, err
	}
	// Scope creation is intentionally a separate statement. Under READ COMMITTED, a
	// concurrent INSERT ... DO NOTHING that waited for the winning transaction
	// is visible to the binding statement below; a same-statement CTE would keep
	// the pre-wait snapshot and could falsely report a conflict.
	if _, err := r.db.Exec(ctx, `
INSERT INTO artifact_scopes (scope_kind, scope_id)
VALUES ($1, $2)
ON CONFLICT DO NOTHING`, scope.kind, scope.id); err != nil {
		discardFreshCandidate()
		if persistencepostgres.SQLState(err) == "55000" {
			return WriteResult{}, fmt.Errorf("create artifact scope: %w", ErrScopeDeleting)
		}
		if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateForeignKeyViolation {
			return WriteResult{}, fmt.Errorf("create artifact scope: %w", ErrInvalidScope)
		}
		return WriteResult{}, fmt.Errorf("create artifact scope: %w", err)
	}
	var storedRevision string
	var mediaType string
	var size int64
	var bindingCreatedAt time.Time
	var revisionCreatedAt time.Time
	var storedObjectKey *string
	err = r.db.QueryRow(ctx, writeSQL,
		scope.kind, scope.id, target.Namespace, target.Name, revision, expectedRevision,
		versionID, digest, blob.Inline, blob.Size, payload.MediaType, string(blob.Backend), nullableBlobKey(blob.Key), reusableKey,
	).Scan(&storedRevision, &mediaType, &size, &bindingCreatedAt, &revisionCreatedAt, &storedObjectKey)
	if errors.Is(err, pgx.ErrNoRows) {
		discardFreshCandidate()
		return WriteResult{}, r.writeConflict(ctx, scope, target, expectedRevision)
	}
	if err != nil {
		switch persistencepostgres.SQLState(err) {
		case "55000":
			discardFreshCandidate()
			return WriteResult{}, fmt.Errorf("write artifact: %w", ErrScopeDeleting)
		case persistencepostgres.SQLStateForeignKeyViolation:
			discardFreshCandidate()
			return WriteResult{}, fmt.Errorf("write artifact: %w", ErrInvalidScope)
		case persistencepostgres.SQLStateUniqueViolation:
			discardFreshCandidate()
			return WriteResult{}, &ConflictError{Ref: target, ExpectedRevision: clone.Pointer(expectedRevision)}
		}
		return WriteResult{}, fmt.Errorf("write artifact %s/%s: %w", target.Namespace, target.Name, err)
	}
	if blob.Backend == BlobFilesystem && storedObjectKey != nil && *storedObjectKey != blob.Key {
		discardUnusedBlob(ctx, r.db, blob)
	}
	exact := storedRevision
	return WriteResult{
		Ref:               ArtifactRef{Namespace: target.Namespace, Name: target.Name, Revision: &exact},
		MediaType:         mediaType,
		Size:              size,
		BindingCreatedAt:  bindingCreatedAt,
		RevisionCreatedAt: revisionCreatedAt,
	}, nil
}

func (r *PostgresRepository) writeConflict(
	ctx context.Context,
	scope Scope,
	target ArtifactRef,
	expectedRevision *string,
) error {
	var frozen bool
	err := r.db.QueryRow(ctx, `
SELECT frozen
FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4`,
		scope.kind, scope.id, target.Namespace, target.Name,
	).Scan(&frozen)
	if err == nil && frozen {
		return ErrArtifactFrozen
	}
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("inspect artifact write conflict: %w", err)
	}
	return &ConflictError{Ref: target, ExpectedRevision: clone.Pointer(expectedRevision)}
}
