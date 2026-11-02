package blob

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync/atomic"

	"github.com/grauwolf32/contractor/internal/apierror"
)

var ErrTransferCapacity = apierror.New(apierror.ArtifactTransferCapacity, "artifact transfer capacity is exhausted")

// ErrRuntimeMissing reports a Server composition defect: an operation
// reached a filesystem payload through a context that does not carry the
// installed blob runtime, for example a fresh context.Background(). It is an
// internal error and proves nothing about the stored bytes, so it is neither
// ErrIntegrity nor ErrMissing.
var ErrRuntimeMissing = errors.New("artifact blob runtime is not installed in the operation context")

// MissingRuntime logs the defect where it is detected, because callers
// map the returned error to a generic internal error.
func MissingRuntime(operation string) error {
	slog.Default().Error("artifact blob runtime is missing from the operation context",
		"operation", operation)
	return fmt.Errorf("%s filesystem artifact blob: %w", operation, ErrRuntimeMissing)
}

// Runtime is installed by Server composition, never by request fields.
// Context propagation keeps transaction-scoped repository construction uniform.
type Runtime struct {
	Store  Storage
	Logger *slog.Logger
	slots  chan struct{}
}
type blobRuntimeKey struct{}
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
func WithRuntime(ctx context.Context, runtime *Runtime) context.Context {
	return context.WithValue(ctx, blobRuntimeKey{}, runtime)
}
func FromContext(ctx context.Context) *Runtime {
	r, _ := ctx.Value(blobRuntimeKey{}).(*Runtime)
	return r
}

// AcquireTransfer is reentrant only within an active outer operation.
func AcquireTransfer(ctx context.Context) (context.Context, func(), error) {
	if err := ctx.Err(); err != nil {
		return ctx, func() {}, err
	}
	r := FromContext(ctx)
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
