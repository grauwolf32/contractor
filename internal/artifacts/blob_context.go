package artifacts

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync/atomic"
	"time"

	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

var ErrTransferCapacity = errors.New("artifact transfer capacity is exhausted")

// errBlobRuntimeMissing reports a Server composition defect: an operation
// reached a filesystem payload through a context that does not carry the
// installed blob runtime, for example a fresh context.Background(). It is an
// internal error and proves nothing about the stored bytes, so it is neither
// ErrArtifactIntegrity nor ErrBlobMissing.
var errBlobRuntimeMissing = errors.New("artifact blob runtime is not installed in the operation context")

// missingBlobRuntime logs the defect where it is detected, because callers
// map the returned error to a generic internal error.
func missingBlobRuntime(operation string) error {
	slog.Default().Error("artifact blob runtime is missing from the operation context",
		"operation", operation)
	return fmt.Errorf("%s filesystem artifact blob: %w", operation, errBlobRuntimeMissing)
}

// PreparePayload lets a write-fenced/transactional caller do filesystem I/O
// before taking its authoritative lock. Unpublished candidates may be orphans
// after cancellation; only registry publication makes them accessible.
func PreparePayload(ctx context.Context, payload Payload) (Payload, error) {
	if err := validatePayload(payload); err != nil {
		return payload, err
	}
	if _, ok := activeBlobStore(ctx).(*FilesystemBlobStore); !ok {
		return payload, nil
	}
	ctx, release, err := AcquireTransfer(ctx)
	if err != nil {
		return payload, err
	}
	defer release()
	object, err := activeBlobStore(ctx).Store(ctx, payload.Data)
	if err != nil {
		return payload, err
	}
	payload.prepared = &object
	return payload, nil
}

func prepareBlob(ctx context.Context, payload Payload) (BlobObject, error) {
	if payload.prepared != nil {
		if err := verifyBlob(*payload.prepared, payload.Data); err != nil {
			return BlobObject{}, err
		}
		return *payload.prepared, nil
	}
	return activeBlobStore(ctx).Store(ctx, payload.Data)
}

// BlobRuntime is installed by Server composition, never by request fields.
// Context propagation keeps transaction-scoped repository construction uniform.
type BlobRuntime struct {
	Store  BlobStore
	Logger *slog.Logger
	slots  chan struct{}
}
type blobRuntimeKey struct{}
type transferKey struct{}
type transferLease struct {
	runtime *BlobRuntime
	active  atomic.Bool
}

func NewBlobRuntime(store BlobStore, logger *slog.Logger) *BlobRuntime {
	if logger == nil {
		logger = slog.Default()
	}
	return &BlobRuntime{Store: store, Logger: logger, slots: make(chan struct{}, 4)}
}
func WithBlobRuntime(ctx context.Context, runtime *BlobRuntime) context.Context {
	return context.WithValue(ctx, blobRuntimeKey{}, runtime)
}
func blobRuntime(ctx context.Context) *BlobRuntime {
	r, _ := ctx.Value(blobRuntimeKey{}).(*BlobRuntime)
	return r
}

// activeBlobStore returns the installed payload backend. Without an installed
// runtime only inline PostgreSQL payloads, whose bytes travel with their
// registry row, are reachable.
func activeBlobStore(ctx context.Context) BlobStore {
	if r := blobRuntime(ctx); r != nil {
		return r.Store
	}
	return runtimelessBlobStore{}
}

// runtimelessBlobStore stores inline payloads, which the registry's backend
// guard rejects on a filesystem installation. Opening a filesystem object
// needs the installation's blob root, so reaching one is a logged internal
// error instead of an integrity failure.
type runtimelessBlobStore struct{ PostgresBlobStore }

func (s runtimelessBlobStore) Read(ctx context.Context, object BlobObject) ([]byte, error) {
	if object.Backend == BlobFilesystem {
		return nil, missingBlobRuntime("read")
	}
	return s.PostgresBlobStore.Read(ctx, object)
}

func (s runtimelessBlobStore) Check(ctx context.Context, object BlobObject) error {
	if object.Backend == BlobFilesystem {
		return missingBlobRuntime("check")
	}
	return s.PostgresBlobStore.Check(ctx, object)
}

// AcquireTransfer is reentrant only within an active outer operation.
func AcquireTransfer(ctx context.Context) (context.Context, func(), error) {
	if err := ctx.Err(); err != nil {
		return ctx, func() {}, err
	}
	r := blobRuntime(ctx)
	if r == nil {
		return ctx, func() {}, nil
	}
	if l, _ := ctx.Value(transferKey{}).(*transferLease); l != nil && l.runtime == r && l.active.Load() {
		return ctx, func() {}, nil
	}
	select {
	case r.slots <- struct{}{}:
	default:
		return ctx, func() {}, ErrTransferCapacity
	}
	l := &transferLease{runtime: r}
	l.active.Store(true)
	return context.WithValue(ctx, transferKey{}, l), func() {
		if l.active.Swap(false) {
			<-r.slots
		}
	}, nil
}

// deletePhysicalAfterCommit schedules the unlink of a filesystem object that
// the caller's transaction leaves unreferenced. Without the installed runtime
// the file could never be unlinked, so the deletion fails before its commit
// instead of silently leaving a physical orphan.
func deletePhysicalAfterCommit(ctx context.Context, tx pgx.Tx, object BlobObject) error {
	if object.Backend != BlobFilesystem {
		return nil
	}
	r := blobRuntime(ctx)
	if r == nil {
		return missingBlobRuntime("delete")
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
func discardUnusedBlob(ctx context.Context, db postgres.DBTX, object BlobObject) {
	if tx, ok := db.(pgx.Tx); ok {
		_ = deletePhysicalAfterCommit(ctx, tx, object)
		return
	}
	discardUnpublishedBlob(ctx, object)
}

// A definitely rejected write never published its fresh candidate, even if
// the caller-owned transaction later rolls back. Remove it immediately.
func discardUnpublishedBlob(ctx context.Context, object BlobObject) {
	if object.Backend != BlobFilesystem {
		return
	}
	r := blobRuntime(ctx)
	if r == nil {
		_ = missingBlobRuntime("discard")
		return
	}
	cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer cancel()
	if err := r.Store.Delete(cleanup, object); err != nil {
		r.Logger.Warn("unused artifact blob deferred to offline cleanup")
	}
}
