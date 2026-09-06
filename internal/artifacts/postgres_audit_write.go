package artifacts

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// WriteAuditArtifact creates a frozen Project binding for trusted
// Controller-generated material. It deliberately has no update path.
func (r *PostgresRepository) WriteAuditArtifact(
	ctx context.Context,
	projectScope Scope,
	target ArtifactRef,
	payload Payload,
) (WriteResult, error) {
	if err := validateScope(projectScope); err != nil || projectScope.kind != ScopeProject {
		return WriteResult{}, ErrInvalidScope
	}
	if target.Revision != nil || !strings.HasPrefix(target.Namespace, "audit-") {
		return WriteResult{}, ErrInvalidName
	}
	if err := validateRef(target); err != nil {
		return WriteResult{}, err
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

	if _, err := r.db.Exec(ctx, `
INSERT INTO artifact_scopes (scope_kind, scope_id)
VALUES ($1, $2)
ON CONFLICT DO NOTHING`, projectScope.kind, projectScope.id); err != nil {
		switch persistencepostgres.SQLState(err) {
		case "55000":
			return WriteResult{}, fmt.Errorf("create Audit artifact scope: %w", ErrScopeDeleting)
		case "23503":
			return WriteResult{}, fmt.Errorf("create Audit artifact scope: %w", ErrInvalidScope)
		default:
			return WriteResult{}, fmt.Errorf("create Audit artifact scope: %w", err)
		}
	}

	var storedRevision, mediaType string
	var size int64
	var bindingCreatedAt, revisionCreatedAt time.Time
	var storedObjectKey *string
	err = r.db.QueryRow(ctx, `
WITH scope_ready AS (
    SELECT 1 FROM artifact_scopes WHERE scope_kind = $1 AND scope_id = $2
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision, frozen
    )
    SELECT $1, $2, $3, $4, $5, true FROM scope_ready
    ON CONFLICT DO NOTHING
    RETURNING created_at
), inserted_blob AS (
    INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend, object_key)
    SELECT $7, $8, $9, $11, $12 FROM created_binding
    ON CONFLICT (sha256) DO UPDATE
    SET sha256 = EXCLUDED.sha256,
        object_key = CASE
            WHEN artifact_blobs.backend = 'filesystem'
             AND artifact_blobs.object_key IS DISTINCT FROM $13::text
            THEN EXCLUDED.object_key ELSE artifact_blobs.object_key END
    WHERE artifact_blobs.backend = EXCLUDED.backend
      AND (artifact_blobs.backend = 'filesystem' OR artifact_blobs.payload = EXCLUDED.payload)
      AND artifact_blobs.size_bytes = EXCLUDED.size_bytes
    RETURNING object_key
), inserted_version AS (
    INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
    SELECT $6, $7, $10 FROM created_binding, inserted_blob
    RETURNING version_id
), inserted_revision AS (
    INSERT INTO artifact_binding_revisions (
        scope_kind, scope_id, namespace, name, revision, version_id
    )
    SELECT $1, $2, $3, $4, $5, version_id FROM inserted_version
    RETURNING revision, created_at
)
SELECT inserted_revision.revision, $10::text, $9::bigint,
       created_binding.created_at, inserted_revision.created_at, (SELECT object_key FROM inserted_blob)
FROM inserted_revision, created_binding`,
		projectScope.kind, projectScope.id, target.Namespace, target.Name,
		revision, versionID, digest, blob.Inline, blob.Size, payload.MediaType, string(blob.Backend), nullableBlobKey(blob.Key), reusableKey,
	).Scan(&storedRevision, &mediaType, &size, &bindingCreatedAt, &revisionCreatedAt, &storedObjectKey)
	if errors.Is(err, pgx.ErrNoRows) {
		return WriteResult{}, &ConflictError{Ref: target}
	}
	if err != nil {
		switch persistencepostgres.SQLState(err) {
		case "55000":
			return WriteResult{}, fmt.Errorf("write Audit artifact: %w", ErrScopeDeleting)
		case "23503":
			return WriteResult{}, fmt.Errorf("write Audit artifact: %w", ErrInvalidScope)
		case "23505":
			return WriteResult{}, &ConflictError{Ref: target}
		default:
			return WriteResult{}, fmt.Errorf("write Audit artifact %s/%s: %w", target.Namespace, target.Name, err)
		}
	}
	if blob.Backend == BlobFilesystem && storedObjectKey != nil && *storedObjectKey != blob.Key {
		discardUnusedBlob(ctx, r.db, blob)
	}
	exact := storedRevision
	return WriteResult{
		Ref:       ArtifactRef{Namespace: target.Namespace, Name: target.Name, Revision: &exact},
		MediaType: mediaType, Size: size,
		BindingCreatedAt: bindingCreatedAt, RevisionCreatedAt: revisionCreatedAt,
	}, nil
}
