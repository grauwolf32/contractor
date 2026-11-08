package postgres

import (
	"context"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// PreparePayload lets a write-fenced/transactional caller do filesystem I/O
// before taking its authoritative lock. Unpublished candidates may be orphans
// after cancellation; only registry publication makes them accessible.
func (s *Storage) PreparePayload(ctx context.Context, db postgres.DBTX, payload artifacts.Payload) (artifacts.Payload, error) {
	if err := artifacts.ValidatePayload(payload); err != nil {
		return payload, err
	}
	ctx, release, err := s.AcquireTransfer(ctx)
	if err != nil {
		return payload, err
	}
	defer release()
	object, err := s.runtime.Store.Store(ctx, payload.Data)
	if err != nil {
		return payload, err
	}
	key, err := s.reusableBlobKey(ctx, db, object)
	if err != nil {
		s.discardUnpublishedBlob(ctx, object)
		return payload, err
	}
	payload.Prepared = blob.NewPrepared(object, key)
	return payload, nil
}

func (s *Storage) prepareBlob(ctx context.Context, payload artifacts.Payload) (blob.Object, error) {
	if payload.Prepared != nil {
		if err := blob.Verify(payload.Prepared.Object(), payload.Data); err != nil {
			return blob.Object{}, err
		}
		return payload.Prepared.Object(), nil
	}
	return s.runtime.Store.Store(ctx, payload.Data)
}

// deletePhysicalAfterCommit schedules the unlink of a filesystem object that
// the caller's transaction leaves unreferenced. The shared runtime keeps the
// adapter alive until the owning transaction and its commit callbacks finish.
func (s *Storage) deletePhysicalAfterCommit(tx pgx.Tx, object blob.Object) {
	if object.Backend != blob.Filesystem {
		return
	}
	r := s.runtime
	if !postgres.AfterCommit(tx, func(cleanup context.Context) {
		if err := r.Store.Delete(cleanup, object); err != nil {
			r.Logger.Warn("artifact blob unlink deferred to offline cleanup")
		}
	}, func(remaining int) {
		r.Logger.Warn("artifact blob cleanup budget exhausted; remaining objects deferred to offline cleanup", "deferred_objects", remaining)
	}) {
		r.Logger.Warn("artifact blob cleanup deferred for externally owned transaction")
	}
	return
}

// discardUnusedBlob removes a fresh candidate that a successful write did not
// attach. A transaction defers deletion until its definite commit; an
// externally owned transaction leaves the candidate for offline cleanup.
func (s *Storage) discardUnusedBlob(ctx context.Context, db postgres.DBTX, object blob.Object) {
	if tx, ok := db.(pgx.Tx); ok {
		s.deletePhysicalAfterCommit(tx, object)
		return
	}
	s.discardUnpublishedBlob(ctx, object)
}

// A definitely rejected write never published its fresh candidate, even if
// the caller-owned transaction later rolls back. Remove it immediately.
func (s *Storage) discardUnpublishedBlob(ctx context.Context, object blob.Object) {
	if object.Backend != blob.Filesystem {
		return
	}
	r := s.runtime
	cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer cancel()
	if err := r.Store.Delete(cleanup, object); err != nil {
		r.Logger.Warn("unused artifact blob deferred to offline cleanup")
	}
}
