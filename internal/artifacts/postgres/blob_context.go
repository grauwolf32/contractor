package postgres

import (
	"context"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	inline "github.com/grauwolf32/contractor/internal/artifacts/blob/backends/postgres"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// PreparePayload lets a write-fenced/transactional caller do filesystem I/O
// before taking its authoritative lock. Unpublished candidates may be orphans
// after cancellation; only registry publication makes them accessible.
func PreparePayload(ctx context.Context, db postgres.DBTX, payload artifacts.Payload) (artifacts.Payload, error) {
	if err := artifacts.ValidatePayload(payload); err != nil {
		return payload, err
	}
	ctx, release, err := blob.AcquireTransfer(ctx)
	if err != nil {
		return payload, err
	}
	defer release()
	object, err := activeBlobStore(ctx).Store(ctx, payload.Data)
	if err != nil {
		return payload, err
	}
	key, err := reusableBlobKey(ctx, db, object)
	if err != nil {
		discardUnpublishedBlob(ctx, object)
		return payload, err
	}
	payload.Prepared = blob.NewPrepared(object, key)
	return payload, nil
}

func prepareBlob(ctx context.Context, payload artifacts.Payload) (blob.Object, error) {
	if payload.Prepared != nil {
		if err := blob.Verify(payload.Prepared.Object(), payload.Data); err != nil {
			return blob.Object{}, err
		}
		return payload.Prepared.Object(), nil
	}
	return activeBlobStore(ctx).Store(ctx, payload.Data)
}

// activeBlobStore returns the installed payload backend. Without an installed
// runtime only inline PostgreSQL payloads, whose bytes travel with their
// registry row, are reachable.
func activeBlobStore(ctx context.Context) blob.Storage {
	if r := blob.FromContext(ctx); r != nil {
		return r.Store
	}
	return runtimelessBlobStore{}
}

// runtimelessBlobStore can only store and open inline payloads. Opening a filesystem object
// needs the installation's blob root, so reaching one is a logged internal
// error instead of an integrity failure.
type runtimelessBlobStore struct{}

func (s runtimelessBlobStore) Read(ctx context.Context, object blob.Object) ([]byte, error) {
	if object.Backend == blob.Filesystem {
		return nil, blob.MissingRuntime("read")
	}
	return (inline.Store{}).Read(ctx, object)
}

func (s runtimelessBlobStore) Check(ctx context.Context, object blob.Object) error {
	if object.Backend == blob.Filesystem {
		return blob.MissingRuntime("check")
	}
	return (inline.Store{}).Check(ctx, object)
}

// deletePhysicalAfterCommit schedules the unlink of a filesystem object that
// the caller's transaction leaves unreferenced. Without the installed runtime
// the file could never be unlinked, so the deletion fails before its commit
// instead of silently leaving a physical orphan.
func deletePhysicalAfterCommit(ctx context.Context, tx pgx.Tx, object blob.Object) error {
	if object.Backend != blob.Filesystem {
		return nil
	}
	r := blob.FromContext(ctx)
	if r == nil {
		return blob.MissingRuntime("delete")
	}
	if !postgres.AfterCommit(tx, func() {
		cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		if err := r.Store.Delete(cleanup, object); err != nil {
			r.Logger.Warn("artifact blob unlink deferred to offline cleanup")
		}
	}) {
		r.Logger.Warn("artifact blob cleanup deferred for externally owned transaction")
	}
	return nil
}

// discardUnusedBlob removes a fresh candidate that a successful write did not
// attach. Only an installed runtime stores filesystem candidates, so a missing
// runtime here is logged and the candidate stays for offline cleanup; the
// write itself succeeded.
func discardUnusedBlob(ctx context.Context, db postgres.DBTX, object blob.Object) {
	if tx, ok := db.(pgx.Tx); ok {
		_ = deletePhysicalAfterCommit(ctx, tx, object)
		return
	}
	discardUnpublishedBlob(ctx, object)
}

// A definitely rejected write never published its fresh candidate, even if
// the caller-owned transaction later rolls back. Remove it immediately.
func discardUnpublishedBlob(ctx context.Context, object blob.Object) {
	if object.Backend != blob.Filesystem {
		return
	}
	r := blob.FromContext(ctx)
	if r == nil {
		_ = blob.MissingRuntime("discard")
		return
	}
	cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer cancel()
	if err := r.Store.Delete(cleanup, object); err != nil {
		r.Logger.Warn("unused artifact blob deferred to offline cleanup")
	}
}

func (runtimelessBlobStore) Store(ctx context.Context, data []byte) (blob.Object, error) {
	return (inline.Store{}).Store(ctx, data)
}
func (runtimelessBlobStore) Delete(ctx context.Context, object blob.Object) error {
	return (inline.Store{}).Delete(ctx, object)
}
