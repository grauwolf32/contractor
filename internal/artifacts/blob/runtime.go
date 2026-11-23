package blob

import (
	"context"
	"log/slog"
	"sync/atomic"

	"github.com/grauwolf32/contractor/internal/apierror"
)

var ErrTransferCapacity = apierror.New(apierror.ArtifactTransferCapacity, "artifact transfer capacity is exhausted")

// Runtime owns one process's physical store and shared transfer admission.
type Runtime struct {
	Store  Storage
	Logger *slog.Logger
	slots  chan struct{}
}
type transferKey struct{}
type transferLease struct {
	runtime *Runtime
	active  atomic.Bool
}

func NewRuntime(store Storage, logger *slog.Logger) *Runtime {
	if logger == nil {
		logger = slog.Default()
	}
	return &Runtime{Store: store, Logger: logger, slots: make(chan struct{}, 4)}
}

// AcquireTransfer is reentrant only within an active outer operation.
func (r *Runtime) AcquireTransfer(ctx context.Context) (context.Context, func(), error) {
	if err := ctx.Err(); err != nil {
		return ctx, func() {}, err
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
