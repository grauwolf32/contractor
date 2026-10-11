package artifacts

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// WriteAuditArtifact creates a frozen Project binding for trusted
// Controller-generated material. It deliberately has no update path; replay
// of an existing binding belongs to Service.WriteAuditArtifact.
func (r *PostgresRepository) WriteAuditArtifact(
	ctx context.Context,
	projectScope Scope,
	target ArtifactRef,
	payload Payload,
) (_ WriteResult, err error) {
	if err := validateScope(projectScope); err != nil || projectScope.kind != ScopeProject {
		return WriteResult{}, ErrInvalidScope
	}
	if target.Revision != nil || !artifactpolicy.IsAuditManagedProjectNamespace(target.Namespace) {
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
	defer func() {
		// As in Write, a definitely refused write never published its fresh
		// candidate; an unclassified statement failure may have.
		if err != nil && payload.prepared == nil && (errors.Is(err, ErrArtifactConflict) ||
			errors.Is(err, ErrScopeDeleting) || errors.Is(err, ErrInvalidScope)) {
			discardUnpublishedBlob(ctx, blob)
		}
	}()
	digest := blob.Digest
	reusableKey, err := reusableBlobKey(ctx, r.db, blob)
	if err != nil {
		return WriteResult{}, err
	}

	if _, err := r.db.Exec(ctx, `
INSERT INTO artifact_scopes (scope_kind, scope_id)
VALUES ($1, $2)
ON CONFLICT DO NOTHING`, projectScope.kind, projectScope.id); err != nil {
		switch persistencepostgres.Classify(err) {
		case persistencepostgres.ClassDeleting:
			return WriteResult{}, fmt.Errorf("create Audit artifact scope: %w", ErrScopeDeleting)
		case persistencepostgres.ClassReferenceViolation:
			return WriteResult{}, fmt.Errorf("create Audit artifact scope: %w", ErrInvalidScope)
		default:
			return WriteResult{}, fmt.Errorf("create Audit artifact scope: %w", err)
		}
	}

	var storedRevision, mediaType string
	var size int64
	var bindingCreatedAt, revisionCreatedAt time.Time
	var storedObjectKey *string
	err = r.db.QueryRow(ctx, writeAuditArtifactSQL,
		projectScope.kind, projectScope.id, target.Namespace, target.Name,
		revision, versionID, digest, blob.Inline, blob.Size, payload.MediaType, string(blob.Backend), nullableBlobKey(blob.Key), reusableKey,
	).Scan(&storedRevision, &mediaType, &size, &bindingCreatedAt, &revisionCreatedAt, &storedObjectKey)
	if errors.Is(err, pgx.ErrNoRows) {
		return WriteResult{}, &ConflictError{Ref: target}
	}
	if err != nil {
		switch persistencepostgres.Classify(err) {
		case persistencepostgres.ClassDeleting:
			return WriteResult{}, fmt.Errorf("write Audit artifact: %w", ErrScopeDeleting)
		case persistencepostgres.ClassReferenceViolation:
			return WriteResult{}, fmt.Errorf("write Audit artifact: %w", ErrInvalidScope)
		case persistencepostgres.ClassDuplicate:
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
