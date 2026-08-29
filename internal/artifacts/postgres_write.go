package artifacts

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"

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
	digest := sha256.Sum256(payload.Data)

	var storedRevision string
	var mediaType string
	var size int64
	err = r.db.QueryRow(ctx, `
WITH inserted_scope AS (
    INSERT INTO artifact_scopes (scope_kind, scope_id)
    VALUES ($1, $2)
    ON CONFLICT DO NOTHING
    RETURNING 1
), scope_ready AS (
    SELECT 1 FROM inserted_scope
    UNION
    SELECT 1 FROM artifact_scopes WHERE scope_kind = $1 AND scope_id = $2
), updated_binding AS (
    UPDATE artifact_bindings
    SET current_revision = $5, updated_at = clock_timestamp()
    WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4
      AND $6::text IS NOT NULL AND current_revision = $6 AND NOT frozen
    RETURNING 1
), created_binding AS (
    INSERT INTO artifact_bindings (
        scope_kind, scope_id, namespace, name, current_revision
    )
    SELECT $1, $2, $3, $4, $5 FROM scope_ready
    WHERE $6::text IS NULL
    ON CONFLICT DO NOTHING
    RETURNING 1
), claimed_binding AS (
    SELECT 1 FROM updated_binding
    UNION ALL
    SELECT 1 FROM created_binding
), inserted_blob AS (
    INSERT INTO artifact_blobs (sha256, payload, size_bytes)
    SELECT $8, $9, $10 FROM claimed_binding
    ON CONFLICT DO NOTHING
    RETURNING 1
), blob_ready AS (
    SELECT 1 FROM inserted_blob
    UNION
    SELECT 1 FROM artifact_blobs, claimed_binding
    WHERE sha256 = $8 AND payload = $9 AND size_bytes = $10
), inserted_version AS (
    INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
    SELECT $7, $8, $11 FROM claimed_binding, blob_ready
    RETURNING version_id
), inserted_revision AS (
    INSERT INTO artifact_binding_revisions (
        scope_kind, scope_id, namespace, name, revision, version_id
    )
    SELECT $1, $2, $3, $4, $5, version_id FROM inserted_version
    RETURNING revision
)
SELECT revision, $11::text, $10::bigint FROM inserted_revision`,
		scope.kind, scope.id, target.Namespace, target.Name, revision, expectedRevision,
		versionID, digest[:], payload.Data, len(payload.Data), payload.MediaType,
	).Scan(&storedRevision, &mediaType, &size)
	if errors.Is(err, pgx.ErrNoRows) {
		return WriteResult{}, r.writeConflict(ctx, scope, target, expectedRevision)
	}
	if err != nil {
		switch persistencepostgres.SQLState(err) {
		case "23503":
			return WriteResult{}, fmt.Errorf("write artifact: %w", ErrInvalidScope)
		case "23505":
			return WriteResult{}, &ConflictError{Ref: target, ExpectedRevision: cloneString(expectedRevision)}
		}
		return WriteResult{}, fmt.Errorf("write artifact %s/%s: %w", target.Namespace, target.Name, err)
	}
	exact := storedRevision
	return WriteResult{
		Ref:       ArtifactRef{Namespace: target.Namespace, Name: target.Name, Revision: &exact},
		MediaType: mediaType, Size: size,
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
