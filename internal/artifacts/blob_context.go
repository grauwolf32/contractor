package artifacts

import (
	"context"
	"errors"
	"log/slog"
	"sync/atomic"
	"time"

	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

var ErrTransferCapacity = errors.New("artifact transfer capacity is exhausted")

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
func activeBlobStore(ctx context.Context) BlobStore {
	if r := blobRuntime(ctx); r != nil {
		return r.Store
	}
	return PostgresBlobStore{}
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

func deletePhysicalAfterCommit(ctx context.Context, tx pgx.Tx, object BlobObject) {
	if object.Backend != BlobFilesystem {
		return
	}
	r := blobRuntime(ctx)
	if r == nil {
		return
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
}

func discardUnusedBlob(ctx context.Context, db postgres.DBTX, object BlobObject) {
	if tx, ok := db.(pgx.Tx); ok {
		deletePhysicalAfterCommit(ctx, tx, object)
		return
	}
	r := blobRuntime(ctx)
	if r == nil {
		return
	}
	cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer cancel()
	if err := r.Store.Delete(cleanup, object); err != nil {
		r.Logger.Warn("unused artifact blob deferred to offline cleanup")
	}
}
