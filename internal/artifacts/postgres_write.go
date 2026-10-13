package artifacts

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/clone"
	"github.com/jackc/pgx/v5"
)

func (r *PostgresRepository) Write(
	ctx context.Context,
	scope Scope,
	target ArtifactRef,
	payload Payload,
	expectedRevision *string,
) (WriteResult, error) {
	return r.write(ctx, scope, target, payload, expectedRevision, false)
}

func (r *PostgresRepository) CreateFrozen(
	ctx context.Context, scope Scope, target ArtifactRef, payload Payload,
) (WriteResult, error) {
	return r.write(ctx, scope, target, payload, nil, true)
}

// write stores payload as a new revision of target. With expectedRevision nil
// it creates the binding, frozen when frozen is set; otherwise it advances the
// unfrozen binding whose current revision is expectedRevision.
func (r *PostgresRepository) write(
	ctx context.Context,
	scope Scope,
	target ArtifactRef,
	payload Payload,
	expectedRevision *string,
	frozen bool,
) (WriteResult, error) {
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
		// same Payload; only this write's fresh candidate is ours to remove.
		if payload.prepared == nil {
			discardUnpublishedBlob(ctx, blob)
		}
	}
	reusableKey, err := reusableBlobKey(ctx, r.db, blob)
	if err != nil {
		discardFreshCandidate()
		return WriteResult{}, err
	}
	if err := r.ensureScope(ctx, scope); err != nil {
		discardFreshCandidate()
		return WriteResult{}, err
	}
	var storedRevision string
	var mediaType string
	var size int64
	var bindingCreatedAt time.Time
	var revisionCreatedAt time.Time
	var storedObjectKey *string
	err = r.db.QueryRow(ctx, writeSQL,
		scope.kind, scope.id, target.Namespace, target.Name, revision, expectedRevision,
		versionID, blob.Digest, blob.Inline, blob.Size, payload.MediaType, string(blob.Backend),
		nullableBlobKey(blob.Key), reusableKey, frozen,
	).Scan(&storedRevision, &mediaType, &size, &bindingCreatedAt, &revisionCreatedAt, &storedObjectKey)
	if errors.Is(err, pgx.ErrNoRows) {
		discardFreshCandidate()
		return WriteResult{}, r.writeConflict(ctx, scope, target, expectedRevision)
	}
	if err != nil {
		err = storeError(fmt.Sprintf("write artifact %s/%s", target.Namespace, target.Name), err,
			&ConflictError{Ref: target, ExpectedRevision: clone.Pointer(expectedRevision)})
		// A definite rejection never published the fresh candidate; an
		// unclassified statement failure may have.
		if definiteRejection(err) {
			discardFreshCandidate()
		}
		return WriteResult{}, err
	}
	if blob.Backend == BlobFilesystem && (storedObjectKey == nil || *storedObjectKey != blob.Key) {
		discardUnusedBlob(ctx, r.db, blob)
	}
	return WriteResult{
		Ref:               artifactExactRef(target.Namespace, target.Name, storedRevision),
		MediaType:         mediaType,
		Size:              size,
		BindingCreatedAt:  bindingCreatedAt,
		RevisionCreatedAt: revisionCreatedAt,
	}, nil
}

// definiteRejection reports a write refusal that storeError classified, after
// which the statement wrote nothing.
func definiteRejection(err error) bool {
	return errors.Is(err, ErrArtifactConflict) || errors.Is(err, ErrRunRevisionLimit) ||
		errors.Is(err, ErrScopeDeleting) || errors.Is(err, ErrInvalidScope)
}

func (r *PostgresRepository) writeConflict(
	ctx context.Context,
	scope Scope,
	target ArtifactRef,
	expectedRevision *string,
) error {
	var revision string
	current := &Metadata{Ref: ArtifactRef{Namespace: target.Namespace, Name: target.Name, Revision: &revision}}
	err := r.db.QueryRow(ctx, `
SELECT current_revision, frozen
FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4`,
		scope.kind, scope.id, target.Namespace, target.Name,
	).Scan(&revision, &current.Frozen)
	if errors.Is(err, pgx.ErrNoRows) {
		current = nil
	} else if err != nil {
		return fmt.Errorf("inspect artifact write conflict: %w", err)
	}
	if refusal := WritePrecondition(target, current, expectedRevision); refusal != nil {
		return refusal
	}
	// The binding moved after the write was refused; the write still lost.
	return &ConflictError{Ref: target, ExpectedRevision: clone.Pointer(expectedRevision)}
}
