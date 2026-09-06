package artifacts

import (
	"context"
	"errors"
	"fmt"
	"time"

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
	digest := blob.Digest
	reusableKey, err := reusableBlobKey(ctx, r.db, blob)
	if err != nil {
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
		if persistencepostgres.SQLState(err) == "55000" {
			return WriteResult{}, fmt.Errorf("create artifact scope: %w", ErrScopeDeleting)
		}
		if persistencepostgres.SQLState(err) == "23503" {
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
	err = r.db.QueryRow(ctx, `
WITH scope_ready AS (
    SELECT 1 FROM artifact_scopes WHERE scope_kind = $1 AND scope_id = $2 FOR KEY SHARE
), updated_binding AS (
    UPDATE artifact_bindings
    SET current_revision = $5, updated_at = clock_timestamp()
    WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4
      AND $6::text IS NOT NULL AND current_revision = $6 AND NOT frozen
      AND EXISTS (SELECT 1 FROM scope_ready)
    RETURNING created_at
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision
    )
    SELECT $1, $2, $3, $4, $5 FROM scope_ready
    WHERE $6::text IS NULL
    ON CONFLICT DO NOTHING
    RETURNING created_at
), claimed_binding AS (
    SELECT created_at FROM updated_binding
    UNION ALL
    SELECT created_at FROM created_binding
), inserted_blob AS (
    INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend, object_key)
    SELECT $8, $9, $10, $12, $13 FROM claimed_binding
    ON CONFLICT (sha256) DO UPDATE
    SET sha256 = EXCLUDED.sha256,
        object_key = CASE
            WHEN artifact_blobs.backend = 'filesystem'
             AND artifact_blobs.object_key IS DISTINCT FROM $14::text
            THEN EXCLUDED.object_key ELSE artifact_blobs.object_key END
    WHERE artifact_blobs.backend = EXCLUDED.backend
      AND (artifact_blobs.backend = 'filesystem' OR artifact_blobs.payload = EXCLUDED.payload)
      AND artifact_blobs.size_bytes = EXCLUDED.size_bytes
    RETURNING object_key
), blob_ready AS (
    SELECT 1 FROM inserted_blob
), inserted_version AS (
    INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
    SELECT $7, $8, $11 FROM claimed_binding, blob_ready
    RETURNING version_id
), inserted_revision AS (
    INSERT INTO artifact_binding_revisions (
        scope_kind, scope_id, namespace, name, revision, version_id
    )
    SELECT $1, $2, $3, $4, $5, version_id FROM inserted_version
    RETURNING revision, created_at
)
SELECT inserted_revision.revision, $11::text, $10::bigint,
       claimed_binding.created_at, inserted_revision.created_at, (SELECT object_key FROM inserted_blob)
FROM inserted_revision, claimed_binding`,
		scope.kind, scope.id, target.Namespace, target.Name, revision, expectedRevision,
		versionID, digest, blob.Inline, blob.Size, payload.MediaType, string(blob.Backend), nullableBlobKey(blob.Key), reusableKey,
	).Scan(&storedRevision, &mediaType, &size, &bindingCreatedAt, &revisionCreatedAt, &storedObjectKey)
	if errors.Is(err, pgx.ErrNoRows) {
		return WriteResult{}, r.writeConflict(ctx, scope, target, expectedRevision)
	}
	if err != nil {
		switch persistencepostgres.SQLState(err) {
		case "55000":
			return WriteResult{}, fmt.Errorf("write artifact: %w", ErrScopeDeleting)
		case "23503":
			return WriteResult{}, fmt.Errorf("write artifact: %w", ErrInvalidScope)
		case "23505":
			return WriteResult{}, &ConflictError{Ref: target, ExpectedRevision: cloneString(expectedRevision)}
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
	return &ConflictError{Ref: target, ExpectedRevision: cloneString(expectedRevision)}
}

func cloneString(value *string) *string {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}
