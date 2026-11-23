// Package claimpass is the claim lifecycle of one pass of a controller that
// leases its subjects through the shared controller claims (PER-R25): the
// Audit controller and the Eval coordinator. A pass handles each claim it
// leased in its own goroutine, joins that handler before it releases the
// claim under a budget of its own, and reports the failures of every claim
// once all of them were released (PER-R26).
//
// The package owns no scheduling policy. Each controller leases its own
// bounded batch, which bounds the goroutines of a pass, gives each handler
// its own operation budget, and decides on its own how to wait, wake or
// retry after the pass.
package claimpass

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// Lifecycle is what a controller supplies for its claims of type C.
type Lifecycle[C any] struct {
	// Handle reconciles one claimed subject under the pass context and
	// reports whether it changed durable state.
	Handle func(context.Context, C) (bool, error)
	// Release gives one claim back. It runs only after Handle returned for
	// that claim, under a context that keeps the pass context's values but
	// not its cancellation, so a pass cancelled by shutdown still releases.
	Release func(context.Context, C) error
	// ReleaseBudget bounds each Release.
	ReleaseBudget time.Duration
	// Lost is the error Release reports for a claim its holder no longer
	// holds: the lease ended and another holder took it, or it was already
	// released. The epoch fences such a holder already, so that release is
	// not a failure of the pass.
	Lost error
}

var errInvalidLifecycle = errors.New("controller claim lifecycle is incomplete")

// Run handles and releases every claim and returns once all of them were
// released. It reports whether any handler changed state, and the handler
// and release failures of every claim, each claim's own joined, in claim
// order. A lifecycle without a handler, release, release budget or lost
// error is a programming error: Run then handles nothing and the claims end
// with their leases.
func Run[C any](ctx context.Context, claims []C, lifecycle Lifecycle[C]) (bool, error) {
	if lifecycle.Handle == nil || lifecycle.Release == nil || lifecycle.ReleaseBudget <= 0 || lifecycle.Lost == nil {
		return false, errInvalidLifecycle
	}
	var changed atomic.Bool
	failures := make([]error, len(claims))
	var wait sync.WaitGroup
	for index, claim := range claims {
		wait.Go(func() {
			worked, handleErr := lifecycle.Handle(ctx, claim)
			if worked {
				changed.Store(true)
			}
			failures[index] = errors.Join(handleErr, lifecycle.release(ctx, claim))
		})
	}
	wait.Wait()
	return changed.Load(), errors.Join(failures...)
}

func (l Lifecycle[C]) release(ctx context.Context, claim C) error {
	return Release(ctx, l.ReleaseBudget, l.Lost, func(releaseContext context.Context) error {
		return l.Release(releaseContext, claim)
	})
}

// Release gives one claim back through release, under a context that keeps
// ctx's values but not its cancellation, bounded by budget. It reports a
// definite failure of release, and nothing when release reports lost: the
// claim's epoch already fences its stale holder (PER-R25, PER-R26). A missing
// release, budget or lost error is a programming error and releases nothing.
func Release(ctx context.Context, budget time.Duration, lost error, release func(context.Context) error) error {
	if release == nil || budget <= 0 || lost == nil {
		return errInvalidLifecycle
	}
	releaseContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), budget)
	defer cancel()
	err := release(releaseContext)
	if errors.Is(err, lost) {
		return nil
	}
	return err
}
